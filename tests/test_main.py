import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import boto3
from botocore.exceptions import ClientError
from main import app
from models.user import UserRegister, UserLogin

client = TestClient(app)

@pytest.fixture
def mock_boto3_client():
    with patch("main.boto3.client") as mock_client:
        mock = MagicMock()
        mock_client.return_value = mock
        mock_client.exceptions = boto3.client('cognito-idp').exceptions
        return mock_client

def test_register_success(mock_boto3_client):
    mock_cognito = mock_boto3_client.return_value
    mock_cognito.sign_up.return_value = {}

    user_data = {
        "username": "testuser",
        "password": "testpass123",
        "email": "test@example.com"
    }

    response = client.post("/register", json=user_data)
    assert response.status_code == 200
    assert response.json() == {"message": "User registered successfully"}
    mock_cognito.sign_up.assert_called_once()

def test_register_user_exists(mock_boto3_client):
    mock_cognito = mock_boto3_client.return_value
    mock_cognito.sign_up.side_effect = mock_boto3_client.exceptions.UsernameExistsException(
        error_response={}, operation_name="SignUp"
    )

    user_data = {
        "username": "existinguser",
        "password": "testpass123",
        "email": "test@example.com"
    }

    response = client.post("/register", json=user_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Username already exists"

def test_login_success(mock_boto3_client):
    mock_cognito = mock_boto3_client.return_value
    mock_cognito.initiate_auth.return_value = {
        "AuthenticationResult": {
            "AccessToken": "fake_access_token",
            "IdToken": "fake_id_token"
        }
    }

    login_data = {
        "username": "testuser",
        "password": "testpass123"
    }

    response = client.post("/login", json=login_data)
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "id_token" in response.json()
    mock_cognito.initiate_auth.assert_called_once()

def test_login_invalid_credentials(mock_boto3_client):
    mock_cognito = mock_boto3_client.return_value
    mock_cognito.initiate_auth.side_effect = mock_boto3_client.exceptions.NotAuthorizedException(
        error_response={}, operation_name="InitiateAuth"
    )

    login_data = {
        "username": "testuser",
        "password": "wrongpass"
    }

    response = client.post("/login", json=login_data)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid credentials"

def test_login_user_not_found(mock_boto3_client):
    mock_cognito = mock_boto3_client.return_value
    mock_cognito.initiate_auth.side_effect = mock_boto3_client.exceptions.UserNotFoundException(
        error_response={}, operation_name="InitiateAuth"
    )

    login_data = {
        "username": "nonexistentuser",
        "password": "testpass123"
    }

    response = client.post("/login", json=login_data)
    assert response.status_code == 404
    assert response.json()["detail"] == "User not found"