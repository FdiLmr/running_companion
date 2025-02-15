import boto3
import os
from dotenv import load_dotenv

load_dotenv()
# Replace with your bucket name
BUCKET_NAME = os.getenv("S3_BUCKET")

def test_s3():
    s3_client = boto3.client("s3")

    # Test file upload
    test_data = {"message": "S3 bucket setup successful!"}
    response = s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key="test.json",
        Body=str(test_data),
        ContentType="application/json"
    )
    print("File uploaded:", response)

    # Test file retrieval
    response = s3_client.get_object(Bucket=BUCKET_NAME, Key="test.json")
    print("File contents:", response["Body"].read().decode("utf-8"))

if __name__ == "__main__":
    test_s3()
