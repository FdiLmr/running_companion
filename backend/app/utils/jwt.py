import requests
from dotenv import load_dotenv
import os
# Load environment variables from .env
load_dotenv()

# Environment variables
COGNITO_REGION = os.getenv("COGNITO_REGION")
USER_POOL_ID = os.getenv("USER_POOL_ID")
APP_CLIENT_ID = os.getenv("APP_CLIENT_ID")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")

# Function to fetch AWS Cognito public keys
def get_cognito_public_keys():
    url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}/.well-known/jwks.json"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Failed to fetch Cognito public keys")
    return response.json().get("keys")
