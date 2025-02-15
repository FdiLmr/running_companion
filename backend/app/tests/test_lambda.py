import json
from strava_update_lambda import lambda_handler

# Simulate an event for testing
mock_event = {
    "test": "data"
}

# Simulate a context (AWS Lambda context object can be mocked if needed)
mock_context = {}

# Call the Lambda function
result = lambda_handler(mock_event, mock_context)
print(json.dumps(result, indent=2))
