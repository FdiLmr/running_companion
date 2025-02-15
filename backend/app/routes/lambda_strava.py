from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordBearer
from routes.user import decode_token  # or any function that yields the user_id
from database.dynamodb import db_save_strava_tokens
import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

strava_router = APIRouter()

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")  # same as user.py

# In a real application, you'd store tokens in a DB or memory.
# This dictionary is purely for demonstration here.
# Key = athlete_id (returned from Strava), value = token info
user_tokens = {}

# @router.get("/strava/connect")
# def connect_to_strava():
#     """
#     Generate the Strava authorization URL with the necessary scopes.
#     """
#     strava_auth_url = (
#         f"https://www.strava.com/oauth/authorize"
#         f"?client_id={STRAVA_CLIENT_ID}"
#         f"&redirect_uri={REDIRECT_URI}"
#         f"&response_type=code"
#         f"&scope=read,profile:read_all,activity:read_all"
#     )
#     return {"auth_url": strava_auth_url}

@strava_router.get("/strava/connect")
def connect_to_strava(token: str = Depends(oauth2_scheme)):
    # Decode token to get user_sub
    decoded = decode_token(token)
    user_id = decoded["sub"]

    strava_auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={STRAVA_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI}"
        f"&response_type=code"
        f"&scope=read,profile:read_all,activity:read_all"
        # Optionally add "state" here for extra security
    )
    return {"auth_url": strava_auth_url}


@strava_router.get("/strava/callback")
def strava_callback(code: str, user_id: str = Depends(get_current_user_id)):
    """
    1) Exchange 'code' for Strava tokens
    2) Save to DB keyed by user_id (via 'state' or a stored session)
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
        raise HTTPException(status_code=400, detail="Failed to exchange code")

    tokens = response.json()
    athlete_id = tokens["athlete"]["id"]

    # (Dev approach) Suppose we stored user_id in state or just test with a single user
    # user_id = "some_hardcoded_user_for_dev"
    # OR parse the 'state' param that Strava sends back to get user_id

    # Save to DB
    db_save_strava_tokens(user_id, str(athlete_id), tokens)

    return {"message": "Strava connected", "athlete_id": athlete_id}

# @router.get("/strava/callback")
# def strava_callback(code: str = Query(...)):
#     token_url = "https://www.strava.com/oauth/token"
#     response = requests.post(
#         token_url,
#         data={
#             "client_id": STRAVA_CLIENT_ID,
#             "client_secret": STRAVA_CLIENT_SECRET,
#             "code": code,
#             "grant_type": "authorization_code",
#         },
#     )

#     if response.status_code != 200:
#         # Print or log the exact error message returned by Strava
#         print("Strava error status:", response.status_code)
#         print("Strava error body:", response.text)

#         # Then raise a more descriptive error
#         raise HTTPException(
#             status_code=400,
#             detail=f"Failed to exchange code for tokens: {response.status_code} - {response.text}",
#         )

#     # ✅ Capture the response JSON in the 'tokens' variable
#     tokens = response.json()
    
#     athlete_id = tokens["athlete"]["id"]

#     # Store locally in a JSON file
#     store_tokens_in_file(athlete_id, tokens)

#     return {"message": "Strava account connected successfully", "tokens": tokens}


# def store_tokens_in_file(athlete_id, tokens):
#     """
#     Append or update local JSON file with the new tokens for this athlete/user.
#     """
#     FILENAME = "strava_tokens.json"
#     try:
#         # Load existing data if file already exists
#         if os.path.exists(FILENAME):
#             with open(FILENAME, "r") as f:
#                 data = json.load(f)
#         else:
#             data = {}

#         # Overwrite or create the entry for this athlete_id
#         data[str(athlete_id)] = {
#             "access_token": tokens["access_token"],
#             "refresh_token": tokens["refresh_token"],
#             "expires_at": tokens["expires_at"]
#         }

#         # Save back to disk
#         with open(FILENAME, "w") as f:
#             json.dump(data, f, indent=2)

#     except Exception as e:
#         print("Error writing tokens to file:", e)