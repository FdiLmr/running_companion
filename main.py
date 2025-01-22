from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from models.user import UserRegister, UserLogin
from routes.strava import router as strava_router
from utils.jwt import get_cognito_public_keys
import os
import boto3
import hmac
import hashlib
import base64
from jose import jwt, JWTError
from jose import jwk

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# Add CORS Middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins (replace with specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Register the Strava router
app.include_router(strava_router)

# Environment variables
COGNITO_REGION = os.getenv("COGNITO_REGION")
USER_POOL_ID = os.getenv("USER_POOL_ID")
APP_CLIENT_ID = os.getenv("APP_CLIENT_ID")
APP_CLIENT_SECRET = os.getenv("APP_CLIENT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")


def calculate_secret_hash(username):
    message = username + APP_CLIENT_ID
    key = bytes(APP_CLIENT_SECRET, "utf-8")
    secret_hash = base64.b64encode(
        hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
    ).decode("utf-8")
    return secret_hash


@app.post("/register")
def register(user: UserRegister):
    client = boto3.client("cognito-idp", region_name=COGNITO_REGION)
    try:
        secret_hash = calculate_secret_hash(user.username)
        client.sign_up(
            ClientId=APP_CLIENT_ID,
            SecretHash=secret_hash,
            Username=user.username,
            Password=user.password,
            UserAttributes=[
                {"Name": "email", "Value": user.email},
            ],
        )
        return {"message": "User registered successfully"}
    except client.exceptions.UsernameExistsException:
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/login")
def login(user: UserLogin):
    client = boto3.client("cognito-idp", region_name=COGNITO_REGION)
    try:
        secret_hash = calculate_secret_hash(user.username)
        response = client.initiate_auth(
            ClientId=APP_CLIENT_ID,
            AuthFlow="USER_PASSWORD_AUTH",
            AuthParameters={
                "USERNAME": user.username,
                "PASSWORD": user.password,
                "SECRET_HASH": secret_hash,
            },
        )
        # Instead of just returning the Access Token,
        # Return the ID Token as well
        id_token = response["AuthenticationResult"]["IdToken"]
        return {
            "access_token": response["AuthenticationResult"]["AccessToken"],
            "id_token": id_token,
        }
    except client.exceptions.NotAuthorizedException:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    except client.exceptions.UserNotFoundException:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


@app.get("/user")
def get_user(token: str = Depends(oauth2_scheme)):
    public_keys = get_cognito_public_keys()
    try:
        header = jwt.get_unverified_header(token)
        kid = header["kid"]

        jwk_data = next((k for k in public_keys if k["kid"] == kid), None)
        if not jwk_data:
            raise HTTPException(status_code=401, detail="Invalid token header")

        # Construct RSA key from the JWK
        rsa_key = jwk.construct(jwk_data, algorithm=JWT_ALGORITHM)

        # Decode and verify the *ID Token* if that's what you're passing to this endpoint
        decoded_token = jwt.decode(
            token,
            rsa_key,
            algorithms=[JWT_ALGORITHM],
            audience=APP_CLIENT_ID,
            issuer=f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}",
        )

        # ID Token usually has 'email' and possibly 'cognito:username'
        return {
            "username": decoded_token.get("cognito:username"),
            "email": decoded_token.get("email"),
        }
    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
