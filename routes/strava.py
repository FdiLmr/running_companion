from fastapi import APIRouter, HTTPException, Query
import requests
import os
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
    """
    Handle the Strava OAuth callback by exchanging the 'code' for an access token
    and refresh token.
    """
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
        raise HTTPException(status_code=400, detail="Failed to exchange code for tokens")

    tokens = response.json()
    athlete_id = tokens["athlete"]["id"]

    # Store tokens in a local dict (for demonstration).
    # In production, you'd store these in a real DB or pass them to your Lambda function as needed.
    user_tokens[athlete_id] = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": tokens["expires_at"],
        "scope": tokens.get("scope", ""),
    }

    return {"message": "Strava account connected successfully", "tokens": tokens}
