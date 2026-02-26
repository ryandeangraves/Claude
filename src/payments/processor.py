"""Payment processing logic."""
import time
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PaymentStatus(Enum):
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    FAILED = "failed"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"


class PaymentError(Exception):
    pass


class InsufficientFundsError(PaymentError):
    pass


class CardDeclinedError(PaymentError):
    pass


@dataclass
class PaymentMethod:
    type: str  # "card", "paypal", "bank_transfer"
    last_four: Optional[str] = None
    expiry_month: Optional[int] = None
    expiry_year: Optional[int] = None
    token: str = field(default_factory=lambda: secrets.token_hex(16))


@dataclass
class Payment:
    id: str
    order_id: str
    amount: float
    currency: str = "USD"
    status: PaymentStatus = PaymentStatus.PENDING
    payment_method: Optional[PaymentMethod] = None
    created_at: float = field(default_factory=time.time)
    refunded_amount: float = 0.0


def validate_card_expiry(month: int, year: int) -> bool:
    now = time.localtime()
    if year < now.tm_year:
        return False
    if year == now.tm_year and month < now.tm_mon:
        return False
    if not 1 <= month <= 12:
        return False
    return True


def validate_payment_amount(amount: float) -> bool:
    return amount > 0 and round(amount, 2) == amount


def authorize_payment(payment: Payment) -> Payment:
    """Simulate payment authorization against a gateway."""
    if payment.status != PaymentStatus.PENDING:
        raise PaymentError("Payment is not in PENDING state")
    if not validate_payment_amount(payment.amount):
        raise PaymentError(f"Invalid payment amount: {payment.amount}")

    pm = payment.payment_method
    if pm and pm.type == "card":
        if pm.last_four == "0002":
            raise InsufficientFundsError("Card declined: insufficient funds")
        if pm.last_four == "0003":
            raise CardDeclinedError("Card declined by issuer")
        if pm.expiry_month and pm.expiry_year:
            if not validate_card_expiry(pm.expiry_month, pm.expiry_year):
                raise PaymentError("Card is expired")

    payment.status = PaymentStatus.AUTHORIZED
    return payment


def capture_payment(payment: Payment) -> Payment:
    if payment.status != PaymentStatus.AUTHORIZED:
        raise PaymentError("Payment must be authorized before capture")
    payment.status = PaymentStatus.CAPTURED
    return payment


def refund_payment(payment: Payment, refund_amount: Optional[float] = None) -> Payment:
    if payment.status not in (PaymentStatus.CAPTURED, PaymentStatus.PARTIALLY_REFUNDED):
        raise PaymentError("Only captured payments can be refunded")

    amount_to_refund = refund_amount if refund_amount is not None else payment.amount
    if amount_to_refund <= 0:
        raise PaymentError("Refund amount must be positive")

    remaining = payment.amount - payment.refunded_amount
    if amount_to_refund > remaining:
        raise PaymentError(
            f"Refund amount {amount_to_refund} exceeds remaining refundable amount {remaining}"
        )

    payment.refunded_amount += amount_to_refund
    if payment.refunded_amount >= payment.amount:
        payment.status = PaymentStatus.REFUNDED
    else:
        payment.status = PaymentStatus.PARTIALLY_REFUNDED

    return payment


def calculate_processing_fee(amount: float, payment_type: str) -> float:
    """Calculate gateway processing fee."""
    fees = {
        "card": 0.029 * amount + 0.30,      # 2.9% + $0.30
        "paypal": 0.0349 * amount + 0.49,   # 3.49% + $0.49
        "bank_transfer": 0.008 * amount,     # 0.8%, capped at $5
    }
    fee = fees.get(payment_type, 0.0)
    if payment_type == "bank_transfer":
        fee = min(fee, 5.0)
    return round(fee, 2)


def get_payment_summary(payment: Payment) -> dict:
    return {
        "id": payment.id,
        "order_id": payment.order_id,
        "amount": payment.amount,
        "currency": payment.currency,
        "status": payment.status.value,
        "refunded_amount": payment.refunded_amount,
        "net_amount": payment.amount - payment.refunded_amount,
    }
