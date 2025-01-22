import boto3
import requests
import time
import json
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("strava_update.log"),
        logging.StreamHandler()
    ]
)

# Environment Variables (set in Lambda configuration or .env)
STRAVA_CLIENT_ID = os.environ.get("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET")
S3_BUCKET = os.environ.get("S3_BUCKET")

# Initial tokens (for testing/demonstration). In production, you’d retrieve these from a database.
INITIAL_ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")     # Example only
INITIAL_REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")   # Example only

# Safe margin to avoid Strava rate-limit (100 requests per 15 minutes)
MAX_ACTIVITY_REQUESTS = 25  

# In-memory dictionary simulating user token storage
# Key is a user_id or athlete_id; value is dict of tokens
user_tokens = {
    "user_id_1": {
        "access_token": INITIAL_ACCESS_TOKEN,
        "refresh_token": INITIAL_REFRESH_TOKEN,
        # A sample expires_at in the past for demonstration (forces refresh).
        # In real usage, set it to the actual expires_at from Strava callback.
        "expires_at": 1672531199,
    },
}

s3_client = boto3.client("s3")

def get_user_tokens(user_id):
    """
    Mock: Retrieve the user tokens for the given user_id from our in-memory dict.
    In production, replace with database or DynamoDB calls.
    """
    return user_tokens.get(user_id)

def update_user_tokens(user_id, tokens):
    """
    Mock: Update the user tokens for the given user_id in our in-memory dict.
    In production, replace with database or DynamoDB calls.
    """
    user_tokens[user_id] = tokens

def refresh_access_token(user_id):
    """
    Refresh Strava access token if expired. Returns a valid (refreshed if needed) access token.
    """
    logging.info(f"Attempting to refresh token for user: {user_id}")
    tokens = get_user_tokens(user_id)

    if not tokens:
        raise ValueError(f"No tokens found for user: {user_id}")

    # If current time is greater than expires_at, token is expired → refresh
    if time.time() > tokens["expires_at"]:
        logging.info(f"Token expired for user {user_id}, refreshing...")
        refresh_url = "https://www.strava.com/oauth/token"
        response = requests.post(
            refresh_url,
            data={
                "client_id": STRAVA_CLIENT_ID,
                "client_secret": STRAVA_CLIENT_SECRET,
                "refresh_token": tokens["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        new_tokens = response.json()
        logging.info(f"Token refresh response for user {user_id}: {new_tokens}")

        # Update our tokens with the new values
        tokens.update(
            {
                "access_token": new_tokens["access_token"],
                "refresh_token": new_tokens["refresh_token"],
                "expires_at": new_tokens["expires_at"],
            }
        )
        update_user_tokens(user_id, tokens)

    return tokens["access_token"]

def fetch_and_save_to_s3(user_id, file_name, data):
    """
    Save fetched data to S3 as a JSON file.
    """
    try:
        key_path = f"user_data/{user_id}/{file_name}"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key_path,
            Body=json.dumps(data),
            ContentType="application/json",
        )
        logging.info(f"Saved {file_name} for user {user_id} to S3 at: {key_path}")
    except Exception as e:
        logging.error(f"Failed to save {file_name} for user {user_id} to S3: {e}")
        raise

def fetch_strava_data(user_id):
    """
    Fetch and save athlete data, zones, stats, and detailed activities for the given user_id.
    """
    logging.info(f"Fetching Strava data for user: {user_id}")
    
    # Get a valid token (refresh if necessary)
    access_token = refresh_access_token(user_id)
    headers = {"Authorization": f"Bearer {access_token}"}
    logging.info(f"Using access token for {user_id}: {access_token}")

    # 1. Fetch Athlete Data
    logging.info("Fetching athlete data...")
    athlete_url = "https://www.strava.com/api/v3/athlete"
    athlete_response = requests.get(athlete_url, headers=headers)
    athlete_response.raise_for_status()
    athlete_data = athlete_response.json()
    fetch_and_save_to_s3(user_id, "athlete_data.json", athlete_data)

    # 2. Fetch Athlete Zones
    try:
        logging.info("Fetching athlete zones...")
        zones_url = "https://www.strava.com/api/v3/athlete/zones"
        zones_response = requests.get(zones_url, headers=headers)
        logging.info(f"Zones response status: {zones_response.status_code}")
        zones_response.raise_for_status()

        athlete_zones = zones_response.json()
        fetch_and_save_to_s3(user_id, "athlete_zones.json", athlete_zones)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching athlete zones for user {user_id}: {e}")

    # 3. Fetch Athlete Stats
    logging.info("Fetching athlete stats...")
    athlete_id = athlete_data.get("id")
    if athlete_id is None:
        logging.warning(f"No valid athlete ID found in athlete_data for user {user_id}.")
    else:
        stats_url = f"https://www.strava.com/api/v3/athletes/{athlete_id}/stats"
        stats_response = requests.get(stats_url, headers=headers)
        stats_response.raise_for_status()
        athlete_stats = stats_response.json()
        fetch_and_save_to_s3(user_id, "athlete_stats.json", athlete_stats)

    # 4. Fetch Detailed Activities
    logging.info("Fetching detailed activities...")
    activities_url = "https://www.strava.com/api/v3/athlete/activities"
    detailed_activities = []
    page = 1

    while len(detailed_activities) < MAX_ACTIVITY_REQUESTS:
        # Fetch a page of activities
        response = requests.get(activities_url, headers=headers, params={"page": page, "per_page": 30})
        response.raise_for_status()

        activities = response.json()
        if not activities:
            # No more activities
            break

        # For each activity, fetch a detailed version
        for activity in activities:
            if len(detailed_activities) >= MAX_ACTIVITY_REQUESTS:
                break

            activity_id = activity["id"]
            detailed_url = f"https://www.strava.com/api/v3/activities/{activity_id}"
            detail_response = requests.get(detailed_url, headers=headers)
            detail_response.raise_for_status()

            detailed_activity = detail_response.json()
            detailed_activities.append(detailed_activity)

            logging.info(f"Fetched detailed data for activity ID: {activity_id}")
            # Rate limit: Pause to avoid hitting Strava limits (100 calls/15min)
            time.sleep(1.5)

        page += 1

    fetch_and_save_to_s3(user_id, "detailed_activities.json", detailed_activities)
    logging.info(f"Finished fetching data for user: {user_id}")

def lambda_handler(event, context):
    """
    AWS Lambda entry point. Iterates over each user in our user_tokens dict and
    fetches their Strava data.
    """
    logging.info("Starting Strava data update...")
    for user_id in user_tokens.keys():
        try:
            fetch_strava_data(user_id)
        except Exception as e:
            logging.error(f"Failed to update data for user {user_id}: {e}")
    logging.info("Finished Strava data update.")
    return {"status": "success"}
