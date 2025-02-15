import os
import boto3

DYNAMODB_REGION = os.getenv("DYNAMODB_REGION", "eu-west-3")
TABLE_NAME = os.getenv("DYNAMODB_TABLE", "strava_tokens")

dynamodb = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
table = dynamodb.Table(TABLE_NAME)

def db_get_all_users():
    """
    Retrieve all items from the strava_tokens table. 
    Each item is expected to have:
      user_id       (PK)
      athlete_id
      access_token
      refresh_token
      expires_at
    """
    resp = table.scan()
    return resp.get("Items", [])

def db_get_strava_tokens(user_id: str):
    """
    Retrieve tokens for a specific user_id from DynamoDB.
    """
    resp = table.get_item(Key={"user_id": user_id})
    return resp.get("Item")

def db_update_strava_tokens(user_id: str, new_values: dict):
    """
    Updates the strava_tokens table with the new values.
    Expects new_values to have 'access_token', 'refresh_token', 'expires_at', 'athlete_id'.
    """
    table.put_item(Item={
        "user_id": user_id,
        "athlete_id": new_values["athlete_id"],
        "access_token": new_values["access_token"],
        "refresh_token": new_values["refresh_token"],
        "expires_at": new_values["expires_at"]
    })