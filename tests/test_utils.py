import pytest
from unittest.mock import patch, MagicMock
from utils.jwt import get_cognito_public_keys
import requests

@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        yield mock_get

def test_get_cognito_public_keys_success(mock_requests_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "keys": [
            {
                "kid": "1234example=",
                "alg": "RS256",
                "kty": "RSA",
                "e": "AQAB",
                "n": "1234567890",
                "use": "sig"
            }
        ]
    }
    mock_requests_get.return_value = mock_response

    keys = get_cognito_public_keys()
    assert isinstance(keys, list)
    assert len(keys) == 1
    assert keys[0]["kid"] == "1234example="
    assert keys[0]["alg"] == "RS256"

def test_get_cognito_public_keys_failure(mock_requests_get):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_requests_get.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        get_cognito_public_keys()
    assert str(exc_info.value) == "Failed to fetch Cognito public keys"