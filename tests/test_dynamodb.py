import os
import boto3
import json
import logging
from botocore.exceptions import ClientError

# Adjust these as needed or set them in your .env / environment
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "strava_tokens")
DYNAMODB_REGION = os.getenv("DYNAMODB_REGION", "eu-west-3")

# Optional: Configure logging to see debug info from boto3
logging.basicConfig(level=logging.INFO)

def main():
    # Print out environment info for debugging
    print("=== DynamoDB Connectivity Test ===")
    print(f"DYNAMODB_REGION: {DYNAMODB_REGION}")
    print(f"DYNAMODB_TABLE:  {DYNAMODB_TABLE}")

    # Initialize a DynamoDB resource
    dynamodb = boto3.resource("dynamodb", region_name=DYNAMODB_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    # Test the AWS account ID (this helps confirm you're on the right account/credentials)
    sts_client = boto3.client("sts")
    try:
        caller_identity = sts_client.get_caller_identity()
        print(f"AWS Account: {caller_identity.get('Account')}")
        print(f"ARN:         {caller_identity.get('Arn')}")
    except ClientError as e:
        print("Error calling STS get_caller_identity (check your credentials):", e)
        return

    # Prepare a test item
    test_user_id = "test_user_sub"
    test_athlete_id = "1234"
    test_tokens = {
        "access_token": "test_access_token",
        "refresh_token": "test_refresh_token",
        "expires_at": 1234567890
    }

    # Put the item
    print("\n--- Putting Item ---")
    try:
        response = table.put_item(
            Item={
                "user_id": test_user_id,
                "athlete_id": test_athlete_id,
                "access_token": test_tokens["access_token"],
                "refresh_token": test_tokens["refresh_token"],
                "expires_at": test_tokens["expires_at"],
            }
        )
        print("PutItem response:", response)
    except ClientError as e:
        print("Failed to put item:", e)
        return

    # Get the item by user_id
    print("\n--- Getting Item by user_id ---")
    try:
        response = table.get_item(Key={"user_id": test_user_id})
        item = response.get("Item")
        print("GetItem response:", response)
        print("Retrieved item:", item)
    except ClientError as e:
        print("Failed to get item:", e)
        return

    # Scan the entire table (for debugging, not recommended in production for big tables)
    print("\n--- Scanning the Table ---")
    try:
        scan_response = table.scan()
        all_items = scan_response.get("Items", [])
        print(f"Scan returned {len(all_items)} items.")
        print(json.dumps(all_items, indent=2, default=str))
    except ClientError as e:
        print("Failed to scan table:", e)
        return

    print("\n=== DynamoDB Test Complete ===\n")

if __name__ == "__main__":
    main()
