import boto3
import requests
import time
import json
import os
import logging
from dotenv import load_dotenv

# Import your DynamoDB helper functions
from database.dynamodb import db_get_all_users, db_get_strava_tokens, db_update_strava_tokens

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

def get_user_tokens(user_id: str):
    """
    Loads Strava tokens from DynamoDB for the given user_id.
    Returns a dict:
      {
        "athlete_id": str,
        "access_token": str,
        "refresh_token": str,
        "expires_at": int
      }
    or None if not found.
    """
    item = db_get_strava_tokens(user_id)
    if not item:
        return None
    
    return {
        "athlete_id": item["athlete_id"],
        "access_token": item["access_token"],
        "refresh_token": item["refresh_token"],
        "expires_at": int(item["expires_at"])  # Ensure it's int if stored as string
    }

def update_user_tokens(user_id: str, updated_tokens: dict):
    """
    Updates Strava tokens in DynamoDB for the given user_id.
    Expects updated_tokens to include "athlete_id", "access_token", "refresh_token", and "expires_at".
    """
    db_update_strava_tokens(user_id, updated_tokens)
    logging.info(f"Updated tokens in DynamoDB for user_id={user_id}")

def refresh_access_token(user_id: str):
    """
    Refresh Strava access token if expired. Returns a valid (refreshed if needed) access token.
    """
    logging.info(f"Attempting to refresh token for user_id: {user_id}")
    tokens = get_user_tokens(user_id)

    if not tokens:
        raise ValueError(f"No tokens found for user_id: {user_id}")

    # If current time is greater than expires_at, token is expired → refresh
    if time.time() > tokens["expires_at"]:
        logging.info(f"Token expired for user_id {user_id}, refreshing...")
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
        logging.info(f"Token refresh response for user_id {user_id}: {new_tokens}")

        # Update local tokens
        tokens["access_token"] = new_tokens["access_token"]
        tokens["refresh_token"] = new_tokens["refresh_token"]
        tokens["expires_at"]   = new_tokens["expires_at"]
        # Retain athlete_id
        # (Strava's response doesn't always return athlete, so we keep the old athlete_id)
        
        update_user_tokens(user_id, tokens)

    return tokens["access_token"]

def fetch_and_save_to_s3(athlete_id, file_name, data):
    """
    Save fetched data to S3 as a JSON file, under prefix: user_data/{athlete_id}/{file_name}
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

def fetch_strava_data(user_id: str):
    """
    Fetches and saves athlete data, zones, stats, and detailed activities for the given user_id.
    1) Loads tokens from DB, refreshes if needed.
    2) Uses athlete_id from tokens to build S3 paths, etc.
    """
    logging.info(f"Fetching Strava data for user_id: {user_id}")

    # Get a valid token (refresh if necessary)
    access_token = refresh_access_token(user_id)
    tokens = get_user_tokens(user_id)  # re-fetch to ensure we have updated
    if not tokens:
        logging.warning(f"No tokens after refresh for user_id: {user_id}. Skipping.")
        return

    athlete_id = tokens["athlete_id"]
    headers = {"Authorization": f"Bearer {access_token}"}
    logging.info(f"Using access token for athlete_id={athlete_id}: {access_token}")

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
    # We can use the retrieved athlete_data["id"] for the stats endpoint:
    retrieved_athlete_id = athlete_data.get("id")
    if not retrieved_athlete_id:
        logging.warning(f"No valid athlete ID found in athlete_data for user_id {user_id}.")
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
            break  # No more activities

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
    AWS Lambda entry point (or local test).
    1) Reads all user tokens from DynamoDB (db_get_all_users()).
    2) For each item, calls fetch_strava_data(user_id).
    """
    logging.info("Starting Strava data update...")

    all_items = db_get_all_users()  # each item has user_id, athlete_id, tokens, etc.
    if not all_items:
        logging.info("No users found in DynamoDB. Nothing to do.")
        return {"status": "success", "details": "No users in table"}

    for item in all_items:
        user_id = item["user_id"]
        try:
            fetch_strava_data(user_id)
        except Exception as e:
            logging.error(f"Failed to update data for user_id {user_id}: {e}")

    logging.info("Finished Strava data update.")
    return {"status": "success"}
