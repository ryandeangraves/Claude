"""Order management business logic."""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class OrderError(Exception):
    pass


@dataclass
class OrderItem:
    product_id: str
    name: str
    quantity: int
    unit_price: float
    discount_percent: float = 0.0

    @property
    def subtotal(self) -> float:
        return self.unit_price * self.quantity * (1 - self.discount_percent / 100)


@dataclass
class Order:
    id: str
    user_id: str
    items: list[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    created_at: float = field(default_factory=time.time)
    shipping_address: Optional[dict] = None
    coupon_code: Optional[str] = None
    tax_rate: float = 0.08

    @property
    def subtotal(self) -> float:
        return sum(item.subtotal for item in self.items)

    @property
    def tax(self) -> float:
        return round(self.subtotal * self.tax_rate, 2)

    @property
    def total(self) -> float:
        return round(self.subtotal + self.tax, 2)


VALID_TRANSITIONS: dict[OrderStatus, list[OrderStatus]] = {
    OrderStatus.PENDING: [OrderStatus.CONFIRMED, OrderStatus.CANCELLED],
    OrderStatus.CONFIRMED: [OrderStatus.PROCESSING, OrderStatus.CANCELLED],
    OrderStatus.PROCESSING: [OrderStatus.SHIPPED, OrderStatus.CANCELLED],
    OrderStatus.SHIPPED: [OrderStatus.DELIVERED],
    OrderStatus.DELIVERED: [OrderStatus.REFUNDED],
    OrderStatus.CANCELLED: [],
    OrderStatus.REFUNDED: [],
}

COUPON_CODES: dict[str, float] = {
    "SAVE10": 10.0,
    "SAVE20": 20.0,
    "FREESHIP": 0.0,  # handled separately
    "HALFOFF": 50.0,
}


def validate_order(order: Order) -> tuple[bool, list[str]]:
    errors = []
    if not order.items:
        errors.append("Order must contain at least one item")
    for item in order.items:
        if item.quantity <= 0:
            errors.append(f"Item '{item.name}' must have a positive quantity")
        if item.unit_price < 0:
            errors.append(f"Item '{item.name}' has a negative price")
        if not 0 <= item.discount_percent <= 100:
            errors.append(f"Item '{item.name}' has an invalid discount")
    if order.shipping_address is None:
        errors.append("Shipping address is required")
    return len(errors) == 0, errors


def apply_coupon(order: Order) -> float:
    """Returns discount amount in dollars."""
    if order.coupon_code is None:
        return 0.0
    discount_pct = COUPON_CODES.get(order.coupon_code.upper())
    if discount_pct is None:
        raise OrderError(f"Invalid coupon code: {order.coupon_code}")
    return round(order.subtotal * discount_pct / 100, 2)


def transition_order(order: Order, new_status: OrderStatus) -> None:
    allowed = VALID_TRANSITIONS.get(order.status, [])
    if new_status not in allowed:
        raise OrderError(
            f"Cannot transition order from {order.status.value} to {new_status.value}"
        )
    order.status = new_status


def cancel_order(order: Order, reason: str = "") -> None:
    if order.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED, OrderStatus.REFUNDED):
        raise OrderError(
            f"Cannot cancel order in '{order.status.value}' status"
        )
    transition_order(order, OrderStatus.CANCELLED)


def calculate_shipping(order: Order, method: str = "standard") -> float:
    subtotal = order.subtotal
    rates = {
        "standard": 5.99 if subtotal < 50 else 0.0,
        "express": 14.99,
        "overnight": 29.99,
    }
    if method not in rates:
        raise OrderError(f"Unknown shipping method: {method}")
    if order.coupon_code and order.coupon_code.upper() == "FREESHIP":
        return 0.0
    return rates[method]


def get_order_summary(order: Order) -> dict:
    return {
        "id": order.id,
        "status": order.status.value,
        "item_count": len(order.items),
        "subtotal": order.subtotal,
        "tax": order.tax,
        "total": order.total,
        "created_at": order.created_at,
    }
