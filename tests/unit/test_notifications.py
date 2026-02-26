"""Unit tests for the notifications service.

Coverage gaps (intentional, to be identified):
- validate_phone_number: not tested at all
- truncate_sms_body: not tested
- render_template: not tested
- send_notification for SMS channel: not tested
- send_notification for PUSH channel: not tested
- retry_failed_notification exceeding max retries: not tested
- build_shipping_notification: not tested
"""
import pytest
from src.notifications.service import (
    Notification,
    NotificationChannel,
    NotificationStatus,
    NotificationError,
    validate_email_address,
    send_notification,
    retry_failed_notification,
    build_order_confirmation_notification,
)


class TestValidateEmailAddress:
    def test_valid_email(self):
        assert validate_email_address("hello@example.com") is True

    def test_invalid_email(self):
        assert validate_email_address("not-an-email") is False

    # Missing: test_edge_cases (same gaps as in test_auth.py)


class TestSendNotification:
    def _make_email_notification(self, recipient="user@example.com"):
        return Notification(
            id="n1",
            user_id="u1",
            channel=NotificationChannel.EMAIL,
            subject="Hello",
            body="Test body",
            recipient=recipient,
        )

    def test_send_email_success(self):
        notif = self._make_email_notification()
        result = send_notification(notif)
        assert result.status == NotificationStatus.SENT
        assert result.sent_at is not None

    def test_send_to_invalid_email_raises(self):
        notif = self._make_email_notification(recipient="not-valid")
        with pytest.raises(NotificationError):
            send_notification(notif)

    def test_send_already_sent_raises(self):
        notif = self._make_email_notification()
        notif.status = NotificationStatus.SENT
        with pytest.raises(NotificationError):
            send_notification(notif)

    def test_send_to_fail_address_marks_failed(self):
        notif = self._make_email_notification(recipient="user@fail.example.com")
        result = send_notification(notif)
        assert result.status == NotificationStatus.FAILED

    # Missing tests:
    # - SMS channel with valid phone sends successfully
    # - SMS channel with invalid phone raises NotificationError
    # - SMS body gets truncated when too long
    # - PUSH channel sends successfully
    # - Subject too long raises NotificationError


class TestRetryNotification:
    def test_retry_failed_notification(self):
        notif = Notification(
            id="n1", user_id="u1",
            channel=NotificationChannel.EMAIL,
            subject="Hi", body="body",
            recipient="user@example.com",
            status=NotificationStatus.FAILED,
        )
        result = retry_failed_notification(notif)
        assert result.status == NotificationStatus.SENT
        assert result.retry_count == 1

    # Missing tests:
    # - retry a non-failed notification raises NotificationError
    # - retry after MAX_RETRIES exceeded raises NotificationError
    # - retry count increments correctly across multiple retries


class TestBuildNotifications:
    def test_build_order_confirmation(self):
        notif = build_order_confirmation_notification(
            user_id="u1",
            email="user@example.com",
            order_id="ORD-001",
            order_total=49.99,
        )
        assert notif.channel == NotificationChannel.EMAIL
        assert "ORD-001" in notif.body
        assert "49.99" in notif.body
        assert notif.recipient == "user@example.com"

    # Missing: test_build_shipping_notification
