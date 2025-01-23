from fastapi import APIRouter, HTTPException, Query
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("STRAVA_REDIRECT_URI")

# In a real application, you'd store tokens in a DB or memory.
# This dictionary is purely for demonstration here.
# Key = athlete_id (returned from Strava), value = token info
user_tokens = {}

@router.get("/strava/connect")
def connect_to_strava():
    """
    Generate the Strava authorization URL with the necessary scopes.
    """
    strava_auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={STRAVA_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=read,profile:read_all,activity:read_all"
    )
    return {"auth_url": strava_auth_url}

@router.get("/strava/callback")
def strava_callback(code: str = Query(...)):
    token_url = "https://www.strava.com/oauth/token"
    response = requests.post(
        token_url,
        data={
            "client_id": STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "code": code,
            "grant_type": "authorization_code",
        },
    )

    if response.status_code != 200:
        # Print or log the exact error message returned by Strava
        print("Strava error status:", response.status_code)
        print("Strava error body:", response.text)

        # Then raise a more descriptive error
        raise HTTPException(
            status_code=400,
            detail=f"Failed to exchange code for tokens: {response.status_code} - {response.text}",
        )

    # ✅ Capture the response JSON in the 'tokens' variable
    tokens = response.json()
    
    athlete_id = tokens["athlete"]["id"]

    # Store locally in a JSON file
    store_tokens_in_file(athlete_id, tokens)

    return {"message": "Strava account connected successfully", "tokens": tokens}


def store_tokens_in_file(athlete_id, tokens):
    """
    Append or update local JSON file with the new tokens for this athlete/user.
    """
    FILENAME = "strava_tokens.json"
    try:
        # Load existing data if file already exists
        if os.path.exists(FILENAME):
            with open(FILENAME, "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Overwrite or create the entry for this athlete_id
        data[str(athlete_id)] = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "expires_at": tokens["expires_at"]
        }

        # Save back to disk
        with open(FILENAME, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print("Error writing tokens to file:", e)