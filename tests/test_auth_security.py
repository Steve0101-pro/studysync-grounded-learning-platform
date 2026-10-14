import uuid

import pytest
from pydantic import ValidationError

from app.schema import LayoutElement, UserCreate, UserLogin, PasswordResetRequest
from app.services.auth_service import create_access_token, decode_access_token, hash_password, verify_password
from app.api.auth import _hash_reset_token


def test_password_hash_and_verify_round_trip():
    password = "StrongPass123"
    hashed = hash_password(password)

    assert hashed != password
    assert hashed.startswith("$2b$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword123", hashed) is False


def test_create_and_decode_access_token_round_trip():
    user_id = uuid.uuid4()
    role = "user"

    token = create_access_token(user_id=user_id, role=role)
    payload = decode_access_token(token)

    assert payload["sub"] == str(user_id)
    assert payload["role"] == role
    assert "exp" in payload


def test_user_create_rejects_weak_password():
    with pytest.raises(ValidationError):
        UserCreate(email="user@example.com", password="weak")

    with pytest.raises(ValidationError):
        UserCreate(email="user@example.com", password="weakpassword")

    with pytest.raises(ValidationError):
        UserCreate(email="user@example.com", password="WEAKPASSWORD")

    user = UserCreate(email="user@example.com", password="StrongPass123")
    assert user.email == "user@example.com"
    assert user.password == "StrongPass123"


def test_auth_email_inputs_are_case_insensitive_and_trimmed():
    assert UserCreate(email="  Student@Example.COM  ", password="StrongPass123").email == "student@example.com"
    assert UserLogin(email="Student@Example.COM ", password="StrongPass123").email == "student@example.com"
    assert PasswordResetRequest(email="Student@Example.COM ").email == "student@example.com"


def test_layout_element_accepts_ocr_confidence_percent_scale():
    model = LayoutElement(
        text="Hello",
        confidence=44.0,
        page_number=1,
        block_number=1,
        paragraph_number=1,
        line_number=1,
        word_number=1,
        bounding_box={"left": 10, "top": 20, "width": 30, "height": 40},
    )

    assert model.confidence == 0.44


def test_password_reset_tokens_are_hashed():
    raw_token = "one-time-recovery-token"
    assert _hash_reset_token(raw_token) != raw_token
    assert _hash_reset_token(raw_token) == _hash_reset_token(raw_token)
