import requests
import time
import json
import boto3
import os
from dotenv import load_dotenv

load_dotenv()
from botocore.exceptions import NoCredentialsError

# AWS S3 Bucket configuration
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
s3_client = boto3.client("s3")

def refresh_access_token(user_id, user_tokens, client_id, client_secret):
    """Refresh Strava access token if expired."""
    refresh_url = "https://www.strava.com/oauth/token"
    tokens = user_tokens[user_id]

    if time.time() > tokens["expires_at"]:  # Token expired
        response = requests.post(
            refresh_url,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": tokens["refresh_token"],
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()

        new_tokens = response.json()
        tokens.update(
            {
                "access_token": new_tokens["access_token"],
                "refresh_token": new_tokens["refresh_token"],
                "expires_at": new_tokens["expires_at"],
            }
        )

    return tokens["access_token"]

def fetch_strava_data(user_id, user_tokens, client_id, client_secret):
    """Fetch all Strava activities incrementally."""
    access_token = refresh_access_token(user_id, user_tokens, client_id, client_secret)
    activities_url = "https://www.strava.com/api/v3/athlete/activities"
    headers = {"Authorization": f"Bearer {access_token}"}

    all_activities = []
    page = 1

    while True:
        response = requests.get(activities_url, headers=headers, params={"page": page, "per_page":30})
        response.raise_for_status()

        activities = response.json()
        if not activities:
            break

        all_activities.extend(activities)
        page += 1

        # Handle Strava rate limits (100 requests per 15 minutes)
        if page % 100 == 0:
            time.sleep(900)

    # Save activities to S3 as a JSON file
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=f"user_data/{user_id}.json",
            Body=json.dumps(all_activities),
            ContentType="application/json",
        )
        print(f"Saved data for user {user_id} to S3.")
    except NoCredentialsError:
        print("AWS credentials not found.")

    return all_activities
