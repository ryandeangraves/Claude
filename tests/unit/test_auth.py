"""Unit tests for auth module.

Coverage gaps (intentional, to be identified):
- validate_password_strength: only happy path tested
- hash_password / verify_password: not tested
- is_account_locked / record_failed_attempt: not tested
- get_session expiry: not tested
- revoke_session: not tested
- require_role: not tested
"""
import pytest
from src.auth.models import (
    User,
    validate_email,
    validate_password_strength,
    create_session,
    get_session,
)


class TestValidateEmail:
    def test_valid_email(self):
        assert validate_email("user@example.com") is True

    def test_valid_email_with_dots(self):
        assert validate_email("first.last@sub.domain.com") is True

    def test_invalid_no_at(self):
        assert validate_email("userexample.com") is False

    def test_invalid_no_domain(self):
        assert validate_email("user@") is False

    def test_empty_string(self):
        assert validate_email("") is False

    # Missing: test_invalid_multiple_at, test_with_special_chars, test_unicode_email


class TestValidatePasswordStrength:
    def test_strong_password(self):
        valid, failures = validate_password_strength("SecureP@ss1")
        assert valid is True
        assert failures == []

    def test_too_short(self):
        valid, failures = validate_password_strength("Ab1!")
        assert valid is False
        assert any("8 characters" in f for f in failures)

    # Missing tests:
    # - password with no uppercase
    # - password with no lowercase
    # - password with no digits
    # - password with no special chars
    # - password that fails multiple rules simultaneously


class TestSessionManagement:
    def test_create_and_get_session(self):
        user = User(id="u1", email="test@example.com", password_hash="hash")
        token = create_session(user)
        session = get_session(token)
        assert session is not None
        assert session["user_id"] == "u1"

    def test_get_nonexistent_session(self):
        assert get_session("nonexistent-token") is None

    # Missing tests:
    # - expired session returns None
    # - revoke_session removes the session
    # - session contains correct role
    # - create_session for admin user
