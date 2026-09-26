"""Function tools exposed to the OpenAI agent.

Each tool wraps existing, already-tested domain code so the agent can only do
things the application already knows how to do.  Tools return plain strings or
dicts, which the SDK serialises for the model.
"""
from agents import function_tool

from src.notifications.service import (
    build_order_confirmation_notification,
    build_shipping_notification,
    validate_email_address,
    validate_phone_number,
)


@function_tool
def check_email_address(email: str) -> str:
    """Check whether a string is a syntactically valid email address.

    Args:
        email: The email address to validate.
    """
    return "valid" if validate_email_address(email) else "invalid"


@function_tool
def check_phone_number(phone: str) -> str:
    """Check whether a string is a valid E.164 phone number (e.g. +14155552671).

    Args:
        phone: The phone number to validate.
    """
    return "valid" if validate_phone_number(phone) else "invalid"


@function_tool
def draft_order_confirmation(user_id: str, email: str, order_id: str, order_total: float) -> dict:
    """Draft (but do not send) the order-confirmation email for an order.

    Args:
        user_id: The customer's user id.
        email: The customer's email address.
        order_id: The order identifier.
        order_total: The order total in dollars.
    """
    n = build_order_confirmation_notification(user_id, email, order_id, order_total)
    return {
        "id": n.id,
        "channel": n.channel.value,
        "recipient": n.recipient,
        "subject": n.subject,
        "body": n.body,
    }


@function_tool
def draft_shipping_sms(user_id: str, phone: str, order_id: str, tracking_number: str) -> dict:
    """Draft (but do not send) the shipping SMS for an order.

    Args:
        user_id: The customer's user id.
        phone: The customer's phone number in E.164 format.
        order_id: The order identifier.
        tracking_number: The carrier tracking number.
    """
    n = build_shipping_notification(user_id, phone, order_id, tracking_number)
    return {
        "id": n.id,
        "channel": n.channel.value,
        "recipient": n.recipient,
        "subject": n.subject,
        "body": n.body,
    }


ALL_TOOLS = [
    check_email_address,
    check_phone_number,
    draft_order_confirmation,
    draft_shipping_sms,
]
