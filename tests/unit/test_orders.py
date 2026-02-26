"""Unit tests for order management.

Coverage gaps (intentional, to be identified):
- OrderItem.subtotal with discount: not tested
- validate_order: only missing-items error tested
- apply_coupon with invalid code: not tested
- cancel_order on shipped/delivered: not tested
- calculate_shipping with FREESHIP coupon: not tested
- calculate_shipping with unknown method: not tested
- get_order_summary: not tested
"""
import pytest
from src.orders.models import (
    Order,
    OrderItem,
    OrderStatus,
    OrderError,
    validate_order,
    apply_coupon,
    transition_order,
    cancel_order,
    calculate_shipping,
)


class TestOrderItem:
    def test_subtotal_no_discount(self):
        item = OrderItem(
            product_id="p1", name="Widget", quantity=3, unit_price=10.00
        )
        assert item.subtotal == 30.00

    # Missing: test_subtotal_with_discount, test_zero_quantity_subtotal


class TestOrderTotals:
    def setup_method(self):
        self.order = Order(
            id="o1",
            user_id="u1",
            items=[
                OrderItem("p1", "Widget", 2, 10.00),
                OrderItem("p2", "Gadget", 1, 25.00),
            ],
            shipping_address={"line1": "123 Main St", "city": "Springfield"},
        )

    def test_subtotal(self):
        assert self.order.subtotal == 45.00

    def test_tax_calculation(self):
        assert self.order.tax == round(45.00 * 0.08, 2)

    def test_total(self):
        expected = round(45.00 + 45.00 * 0.08, 2)
        assert self.order.total == expected


class TestValidateOrder:
    def test_valid_order(self):
        order = Order(
            id="o1",
            user_id="u1",
            items=[OrderItem("p1", "Widget", 1, 10.00)],
            shipping_address={"line1": "123 Main St"},
        )
        valid, errors = validate_order(order)
        assert valid is True
        assert errors == []

    def test_empty_items(self):
        order = Order(id="o1", user_id="u1", shipping_address={"line1": "x"})
        valid, errors = validate_order(order)
        assert valid is False
        assert any("at least one item" in e for e in errors)

    # Missing tests:
    # - negative item price
    # - zero quantity item
    # - discount out of range (> 100)
    # - missing shipping address
    # - multiple validation errors at once


class TestOrderTransitions:
    def test_pending_to_confirmed(self):
        order = Order(id="o1", user_id="u1")
        transition_order(order, OrderStatus.CONFIRMED)
        assert order.status == OrderStatus.CONFIRMED

    def test_invalid_transition(self):
        order = Order(id="o1", user_id="u1", status=OrderStatus.DELIVERED)
        with pytest.raises(OrderError):
            transition_order(order, OrderStatus.CONFIRMED)

    def test_cancel_pending_order(self):
        order = Order(id="o1", user_id="u1")
        cancel_order(order)
        assert order.status == OrderStatus.CANCELLED

    # Missing tests:
    # - cancel confirmed order (should succeed)
    # - cancel shipped order (should raise OrderError)
    # - cancel delivered order (should raise OrderError)
    # - full happy-path transition chain


class TestCouponApplication:
    def test_no_coupon(self):
        order = Order(
            id="o1", user_id="u1",
            items=[OrderItem("p1", "Widget", 1, 100.00)],
        )
        assert apply_coupon(order) == 0.0

    def test_valid_coupon_save10(self):
        order = Order(
            id="o1", user_id="u1",
            items=[OrderItem("p1", "Widget", 1, 100.00)],
            coupon_code="SAVE10",
        )
        assert apply_coupon(order) == 10.0

    # Missing tests:
    # - SAVE20, HALFOFF coupons
    # - FREESHIP (discount is 0, but shipping becomes free)
    # - case-insensitive coupon code matching
    # - invalid coupon code raises OrderError


class TestCalculateShipping:
    def test_standard_below_threshold(self):
        order = Order(
            id="o1", user_id="u1",
            items=[OrderItem("p1", "Widget", 1, 30.00)],
        )
        assert calculate_shipping(order, "standard") == 5.99

    def test_standard_above_threshold_free(self):
        order = Order(
            id="o1", user_id="u1",
            items=[OrderItem("p1", "Widget", 1, 60.00)],
        )
        assert calculate_shipping(order, "standard") == 0.0

    # Missing tests:
    # - express shipping rate
    # - overnight shipping rate
    # - FREESHIP coupon makes shipping free regardless of method
    # - unknown shipping method raises OrderError
