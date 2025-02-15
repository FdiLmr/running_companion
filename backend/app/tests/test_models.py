import pytest
from pydantic import BaseModel, ValidationError, EmailStr
from models.user import UserRegister, UserLogin

def test_valid_user_register():
    user_data = {
        "username": "testuser",
        "password": "testpass123",
        "email": "test@example.com"
    }
    user = UserRegister(**user_data)
    assert user.username == user_data["username"]
    assert user.password == user_data["password"]
    assert user.email == user_data["email"]

def test_invalid_user_register():
    # Test with missing required field
    with pytest.raises(ValidationError) as exc_info:
        UserRegister(username="test", password="test")
    assert "email" in str(exc_info.value)

    # Test with empty username
    with pytest.raises(ValidationError) as exc_info:
        UserRegister(username="", password="test", email="test@example.com")
    assert "username" in str(exc_info.value)

    # Test with empty password
    with pytest.raises(ValidationError) as exc_info:
        UserRegister(username="test", password="", email="test@example.com")
    assert "password" in str(exc_info.value)

def test_valid_user_login():
    login_data = {
        "username": "testuser",
        "password": "testpass123"
    }
    user = UserLogin(**login_data)
    assert user.username == login_data["username"]
    assert user.password == login_data["password"]

def test_invalid_user_login():
    # Test with missing required field
    with pytest.raises(ValidationError) as exc_info:
        UserLogin(username="test")
    assert "password" in str(exc_info.value)

    # Test with empty username
    with pytest.raises(ValidationError) as exc_info:
        UserLogin(username="", password="test")
    assert "username" in str(exc_info.value)

    # Test with empty password
    with pytest.raises(ValidationError) as exc_info:
        UserLogin(username="test", password="")
    assert "password" in str(exc_info.value)