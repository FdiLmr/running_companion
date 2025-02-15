from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from utils.jwt import get_cognito_public_keys
from jose import jwt
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the router
user_router = APIRouter()

# OAuth2 scheme for extracting the token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Load configuration from environment variables
COGNITO_REGION = os.getenv("COGNITO_REGION")
USER_POOL_ID = os.getenv("USER_POOL_ID")
APP_CLIENT_ID = os.getenv("APP_CLIENT_ID")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "RS256")

# Decode the token using Cognito public keys
def decode_token(token: str):
    public_keys = get_cognito_public_keys()
    try:
        # Extract token header and find the corresponding key
        header = jwt.get_unverified_header(token)
        key = next((k for k in public_keys if k["kid"] == header["kid"]), None)
        if not key:
            raise HTTPException(status_code=401, detail="Invalid token header")

        # Decode the token
        decoded_token = jwt.decode(
            token,
            jwt.algorithms.RSAAlgorithm.from_jwk(key),
            algorithms=[JWT_ALGORITHM],
            audience=APP_CLIENT_ID,
            issuer=f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}",
        )
        return decoded_token
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# Endpoint: Retrieve user information
@user_router.get("/user")
def get_user(token: str = Depends(oauth2_scheme)):
    decoded_token = decode_token(token)
    user_sub = decoded_token.get("sub")  # The unique Cognito user identifier
    username = decoded_token.get("username", "Unknown")
    email = decoded_token.get("email", "Unknown")

    return {
        "user_sub": user_sub,
        "username": username,
        "email": email,
    }
