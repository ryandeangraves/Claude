"""Notification service for sending emails and SMS."""
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NotificationChannel(Enum):
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


class NotificationStatus(Enum):
    QUEUED = "queued"
    SENT = "sent"
    FAILED = "failed"
    BOUNCED = "bounced"


class NotificationError(Exception):
    pass


@dataclass
class Notification:
    id: str
    user_id: str
    channel: NotificationChannel
    subject: str
    body: str
    recipient: str  # email address or phone number
    status: NotificationStatus = NotificationStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    retry_count: int = 0


MAX_RETRIES = 3
SMS_MAX_LENGTH = 160
EMAIL_SUBJECT_MAX_LENGTH = 998


def validate_email_address(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def validate_phone_number(phone: str) -> bool:
    """Validates E.164 format phone numbers."""
    pattern = r"^\+[1-9]\d{1,14}$"
    return bool(re.match(pattern, phone))


def truncate_sms_body(body: str) -> str:
    if len(body) <= SMS_MAX_LENGTH:
        return body
    return body[: SMS_MAX_LENGTH - 3] + "..."


def render_template(template: str, context: dict) -> str:
    """Simple template rendering with {{variable}} syntax."""
    result = template
    for key, value in context.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


def send_notification(notification: Notification) -> Notification:
    """Simulate sending a notification through the appropriate channel."""
    if notification.status == NotificationStatus.SENT:
        raise NotificationError("Notification has already been sent")

    if notification.channel == NotificationChannel.EMAIL:
        if not validate_email_address(notification.recipient):
            raise NotificationError(f"Invalid email address: {notification.recipient}")
        if len(notification.subject) > EMAIL_SUBJECT_MAX_LENGTH:
            raise NotificationError("Email subject is too long")

    elif notification.channel == NotificationChannel.SMS:
        if not validate_phone_number(notification.recipient):
            raise NotificationError(f"Invalid phone number: {notification.recipient}")
        notification.body = truncate_sms_body(notification.body)

    elif notification.channel == NotificationChannel.PUSH:
        pass  # Push notifications validated by the push service

    # Simulate occasional failures for retry testing
    if notification.recipient.endswith("@fail.example.com"):
        notification.status = NotificationStatus.FAILED
        return notification

    notification.status = NotificationStatus.SENT
    notification.sent_at = time.time()
    return notification


def retry_failed_notification(notification: Notification) -> Notification:
    if notification.status != NotificationStatus.FAILED:
        raise NotificationError("Only failed notifications can be retried")
    if notification.retry_count >= MAX_RETRIES:
        raise NotificationError(
            f"Notification has exceeded maximum retries ({MAX_RETRIES})"
        )
    notification.retry_count += 1
    return send_notification(notification)


def build_order_confirmation_notification(
    user_id: str,
    email: str,
    order_id: str,
    order_total: float,
) -> Notification:
    return Notification(
        id=f"notif_{order_id}",
        user_id=user_id,
        channel=NotificationChannel.EMAIL,
        subject=f"Order Confirmation #{order_id}",
        body=render_template(
            "Thank you for your order! Order #{{order_id}} has been confirmed. "
            "Total: ${{total}}. We'll notify you when it ships.",
            {"order_id": order_id, "total": f"{order_total:.2f}"},
        ),
        recipient=email,
    )


def build_shipping_notification(
    user_id: str,
    phone: str,
    order_id: str,
    tracking_number: str,
) -> Notification:
    return Notification(
        id=f"notif_ship_{order_id}",
        user_id=user_id,
        channel=NotificationChannel.SMS,
        subject="Your order has shipped",
        body=render_template(
            "Your order #{{order_id}} has shipped! Track it: {{tracking}}",
            {"order_id": order_id, "tracking": tracking_number},
        ),
        recipient=phone,
    )
