"""Unit tests for payments processor.

Coverage gaps (intentional, to be identified):
- validate_card_expiry: only tested partially
- InsufficientFundsError / CardDeclinedError paths: not tested
- capture_payment on non-authorized payment: not tested
- partial refund: not tested
- refund exceeding available amount: not tested
- calculate_processing_fee for bank_transfer cap: not tested
- get_payment_summary: not tested
"""
import pytest
from src.payments.processor import (
    Payment,
    PaymentMethod,
    PaymentStatus,
    PaymentError,
    InsufficientFundsError,
    CardDeclinedError,
    validate_card_expiry,
    validate_payment_amount,
    authorize_payment,
    capture_payment,
    refund_payment,
    calculate_processing_fee,
)


class TestValidateCardExpiry:
    def test_future_year_is_valid(self):
        assert validate_card_expiry(1, 2099) is True

    def test_past_year_is_invalid(self):
        assert validate_card_expiry(1, 2020) is False

    # Missing tests:
    # - current month/year edge case
    # - past month in current year
    # - invalid month (0 or 13)


class TestValidatePaymentAmount:
    def test_valid_amount(self):
        assert validate_payment_amount(99.99) is True

    def test_zero_amount(self):
        assert validate_payment_amount(0.0) is False

    def test_negative_amount(self):
        assert validate_payment_amount(-10.00) is False

    # Missing: test for amounts with more than 2 decimal places


class TestAuthorizePayment:
    def test_successful_authorization(self):
        payment = Payment(
            id="pay1", order_id="o1", amount=50.00,
            payment_method=PaymentMethod(type="card", last_four="1234"),
        )
        result = authorize_payment(payment)
        assert result.status == PaymentStatus.AUTHORIZED

    def test_already_authorized_raises(self):
        payment = Payment(
            id="pay1", order_id="o1", amount=50.00,
            status=PaymentStatus.AUTHORIZED,
        )
        with pytest.raises(PaymentError):
            authorize_payment(payment)

    # Missing tests:
    # - last_four "0002" triggers InsufficientFundsError
    # - last_four "0003" triggers CardDeclinedError
    # - expired card raises PaymentError
    # - invalid payment amount raises PaymentError


class TestCapturePayment:
    def test_capture_authorized_payment(self):
        payment = Payment(
            id="pay1", order_id="o1", amount=50.00,
            status=PaymentStatus.AUTHORIZED,
        )
        result = capture_payment(payment)
        assert result.status == PaymentStatus.CAPTURED

    # Missing: test_capture_non_authorized_raises


class TestRefundPayment:
    def test_full_refund(self):
        payment = Payment(
            id="pay1", order_id="o1", amount=100.00,
            status=PaymentStatus.CAPTURED,
        )
        result = refund_payment(payment)
        assert result.status == PaymentStatus.REFUNDED
        assert result.refunded_amount == 100.00

    # Missing tests:
    # - partial refund sets PARTIALLY_REFUNDED status
    # - second partial refund that completes the refund
    # - refund amount exceeds remaining raises PaymentError
    # - refund on PENDING payment raises PaymentError
    # - refund of zero or negative amount raises PaymentError


class TestCalculateProcessingFee:
    def test_card_fee(self):
        fee = calculate_processing_fee(100.00, "card")
        assert fee == round(0.029 * 100 + 0.30, 2)

    def test_paypal_fee(self):
        fee = calculate_processing_fee(100.00, "paypal")
        assert fee == round(0.0349 * 100 + 0.49, 2)

    # Missing tests:
    # - bank_transfer fee below cap
    # - bank_transfer fee at exactly the cap
    # - bank_transfer fee above cap (should be capped at $5.00)
    # - unknown payment type returns 0.0
