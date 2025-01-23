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

# Safe margin to avoid Strava rate-limit (100 requests per 15 minutes)
MAX_ACTIVITY_REQUESTS = 25

s3_client = boto3.client("s3")

def get_user_tokens(athlete_id: str):
    """
    Reads from 'strava_tokens.json' the tokens for the given athlete_id.
    Returns a dict of { access_token, refresh_token, expires_at } or None if not found.
    """
    if not os.path.exists("strava_tokens.json"):
        return None

    with open("strava_tokens.json", "r") as f:
        data = json.load(f)

    return data.get(athlete_id)  # athlete_id is a string key in the JSON

def update_user_tokens(athlete_id: str, updated_tokens: dict):
    """
    Updates the 'strava_tokens.json' file with the new tokens for the given athlete_id.
    """
    FILENAME = "strava_tokens.json"
    if os.path.exists(FILENAME):
        with open(FILENAME, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[athlete_id] = {
        "access_token": updated_tokens["access_token"],
        "refresh_token": updated_tokens["refresh_token"],
        "expires_at": updated_tokens["expires_at"]
    }

    with open(FILENAME, "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Updated tokens in {FILENAME} for athlete_id={athlete_id}")

def refresh_access_token(athlete_id: str):
    """
    Refresh Strava access token if expired. Returns a valid (refreshed if needed) access token.
    """
    logging.info(f"Attempting to refresh token for athlete: {athlete_id}")
    tokens = get_user_tokens(athlete_id)

    if not tokens:
        raise ValueError(f"No tokens found for athlete: {athlete_id}")

    # Check expiration
    if time.time() > tokens["expires_at"]:
        logging.info(f"Token expired for athlete {athlete_id}, refreshing...")
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
        logging.info(f"Token refresh response for athlete {athlete_id}: {new_tokens}")

        # Update local tokens
        tokens["access_token"] = new_tokens["access_token"]
        tokens["refresh_token"] = new_tokens["refresh_token"]
        tokens["expires_at"]   = new_tokens["expires_at"]

        update_user_tokens(athlete_id, tokens)

    return tokens["access_token"]

def fetch_and_save_to_s3(athlete_id, file_name, data):
    """
    Save fetched data to S3 as a JSON file.
    """
    try:
        key_path = f"user_data/{athlete_id}/{file_name}"
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key_path,
            Body=json.dumps(data),
            ContentType="application/json",
        )
        logging.info(f"Saved {file_name} for athlete {athlete_id} to S3 at: {key_path}")
    except Exception as e:
        logging.error(f"Failed to save {file_name} for athlete {athlete_id} to S3: {e}")
        raise

def fetch_strava_data(athlete_id: str):
    """
    Fetch and save athlete data, zones, stats, and detailed activities for the given athlete_id.
    """
    logging.info(f"Fetching Strava data for athlete: {athlete_id}")

    # Get a valid token (refresh if necessary)
    access_token = refresh_access_token(athlete_id)
    headers = {"Authorization": f"Bearer {access_token}"}
    logging.info(f"Using access token for {athlete_id}: {access_token}")

    # 1. Fetch Athlete Data
    logging.info("Fetching athlete data...")
    athlete_url = "https://www.strava.com/api/v3/athlete"
    athlete_response = requests.get(athlete_url, headers=headers)
    athlete_response.raise_for_status()
    athlete_data = athlete_response.json()
    fetch_and_save_to_s3(athlete_id, "athlete_data.json", athlete_data)

    # 2. Fetch Athlete Zones
    try:
        logging.info("Fetching athlete zones...")
        zones_url = "https://www.strava.com/api/v3/athlete/zones"
        zones_response = requests.get(zones_url, headers=headers)
        logging.info(f"Zones response status: {zones_response.status_code}")
        zones_response.raise_for_status()
        athlete_zones = zones_response.json()
        fetch_and_save_to_s3(athlete_id, "athlete_zones.json", athlete_zones)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching athlete zones for athlete {athlete_id}: {e}")

    # 3. Fetch Athlete Stats
    logging.info("Fetching athlete stats...")
    retrieved_athlete_id = athlete_data.get("id")
    if retrieved_athlete_id is None:
        logging.warning(f"No valid athlete ID found in athlete_data for athlete {athlete_id}.")
    else:
        stats_url = f"https://www.strava.com/api/v3/athletes/{retrieved_athlete_id}/stats"
        stats_response = requests.get(stats_url, headers=headers)
        stats_response.raise_for_status()
        athlete_stats = stats_response.json()
        fetch_and_save_to_s3(athlete_id, "athlete_stats.json", athlete_stats)

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
            # Rate limit: Pause to avoid hitting Strava's 100 calls/15 min limit
            time.sleep(1.5)

        page += 1

    fetch_and_save_to_s3(athlete_id, "detailed_activities.json", detailed_activities)
    logging.info(f"Finished fetching data for athlete: {athlete_id}")

def lambda_handler(event, context):
    """
    AWS Lambda entry point (here, local test).
    1) Reads the strava_tokens.json file to see which athlete_ids we have.
    2) For each athlete_id, calls fetch_strava_data.
    """
    logging.info("Starting Strava data update...")

    if not os.path.exists("strava_tokens.json"):
        logging.error("No 'strava_tokens.json' file found. Cannot update.")
        return {"status": "failure", "reason": "No token file"}

    with open("strava_tokens.json", "r") as f:
        tokens_data = json.load(f)

    # Iterate over each athlete_id in the tokens file
    for athlete_id in tokens_data.keys():
        try:
            fetch_strava_data(athlete_id)
        except Exception as e:
            logging.error(f"Failed to update data for athlete_id {athlete_id}: {e}")

    logging.info("Finished Strava data update.")
    return {"status": "success"}
