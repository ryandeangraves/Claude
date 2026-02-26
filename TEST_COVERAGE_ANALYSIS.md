# Test Coverage Analysis

**Generated:** 2026-02-26
**Test runner:** pytest + pytest-cov
**Total coverage:** 81% (268/332 statements)

---

## Summary Table

| Module | Coverage | Statements | Missed | Risk |
|--------|----------|------------|--------|------|
| `src/auth/models.py` | **68%** | 81 | 26 | 🔴 High |
| `src/payments/processor.py` | **84%** | 89 | 14 | 🟠 Medium |
| `src/notifications/service.py` | **81%** | 77 | 15 | 🟠 Medium |
| `src/orders/models.py` | **89%** | 85 | 9 | 🟡 Low-Medium |
| **TOTAL** | **81%** | **332** | **64** | |

There are **0 integration tests** and **no test coverage for error recovery flows**.

---

## 1. `src/auth/models.py` — 68% — 🔴 Highest Priority

Authentication is the highest-risk module in any application. A bug here can lead
to account compromise, privilege escalation, or data breaches.

### Untested code (26 lines)

| Lines | Function | What's missing |
|-------|----------|----------------|
| 53–59 | `hash_password` / `verify_password` | Entire password hashing pipeline is untested |
| 64–69 | `verify_password` | Password comparison (including timing-safe `compare_digest`) |
| 73–74 | `is_account_locked` | Lockout logic never executed in tests |
| 78–82 | `record_failed_attempt` | Failed login recording never tested |
| 86 | `record_failed_attempt` | Pruning of stale attempt timestamps |
| 105–106 | `create_session` | Admin/moderator session roles |
| 111–114 | `get_session` | **Session expiry path** — expired tokens are never cleaned up in tests |
| 118–121 | `revoke_session` / `require_role` | Logout and role enforcement completely untested |

### Proposed tests to add

```python
# tests/unit/test_auth.py

class TestPasswordHashing:
    def test_hash_password_produces_consistent_result(self):
        hashed, salt = hash_password("MyP@ss1")
        hashed2, _ = hash_password("MyP@ss1", salt)
        assert hashed == hashed2

    def test_different_passwords_produce_different_hashes(self):
        h1, salt = hash_password("Password1!")
        h2, _ = hash_password("Password2!", salt)
        assert h1 != h2

    def test_verify_password_correct(self):
        hashed, salt = hash_password("SecureP@ss1")
        assert verify_password("SecureP@ss1", hashed, salt) is True

    def test_verify_password_wrong(self):
        hashed, salt = hash_password("SecureP@ss1")
        assert verify_password("WrongPassword!", hashed, salt) is False


class TestPasswordStrengthAllRules:
    """Each rule must be independently tested."""
    @pytest.mark.parametrize("password,expected_fragment", [
        ("nouppercase1!", "uppercase"),
        ("NOLOWERCASE1!", "lowercase"),
        ("NoDigits!", "digit"),
        ("NoSpecial1", "special character"),
        ("Ab1!", "8 characters"),
    ])
    def test_each_rule_independently(self, password, expected_fragment):
        valid, failures = validate_password_strength(password)
        assert valid is False
        assert any(expected_fragment in f for f in failures)


class TestAccountLockout:
    def test_account_not_locked_initially(self):
        assert is_account_locked("new_user") is False

    def test_account_locked_after_max_attempts(self):
        for _ in range(MAX_FAILED_ATTEMPTS):
            record_failed_attempt("lock_test_user")
        assert is_account_locked("lock_test_user") is True

    def test_lockout_expires_after_duration(self):
        # Use time mocking to simulate elapsed lockout period
        ...


class TestSessionExpiry:
    def test_expired_session_returns_none(self):
        user = User(id="u99", email="e@e.com", password_hash="h")
        token = create_session(user)
        # Manually expire the session
        from src.auth.models import _sessions
        _sessions[token]["expires_at"] = time.time() - 1
        assert get_session(token) is None

    def test_revoke_session(self):
        user = User(id="u100", email="e@e.com", password_hash="h")
        token = create_session(user)
        assert revoke_session(token) is True
        assert get_session(token) is None

    def test_revoke_nonexistent_session(self):
        assert revoke_session("ghost-token") is False


class TestRequireRole:
    def test_admin_can_access_admin_route(self):
        session = {"role": "admin"}
        assert require_role(session, "admin") is True

    def test_user_cannot_access_admin_route(self):
        session = {"role": "user"}
        assert require_role(session, "admin") is False

    def test_moderator_can_access_user_route(self):
        session = {"role": "moderator"}
        assert require_role(session, "user") is True
```

---

## 2. `src/payments/processor.py` — 84% — 🟠 Medium Priority

Payment failures and refund edge cases are where bugs have direct financial impact.

### Untested code (14 lines)

| Lines | Function | What's missing |
|-------|----------|----------------|
| 56, 58 | `validate_card_expiry` | Same month/year edge case; invalid month (0, 13) |
| 71, 76, 78 | `authorize_payment` | `InsufficientFundsError` and `CardDeclinedError` paths |
| 80–81 | `authorize_payment` | Expired card raises `PaymentError` |
| 89 | `authorize_payment` | Invalid amount raises `PaymentError` |
| 96 | `capture_payment` | Capture of non-authorized payment raises `PaymentError` |
| 100, 104 | `refund_payment` | Partial refund / refund exceeds remaining amount |
| 112 | `refund_payment` | Zero/negative refund amount |
| 126, 131 | `calculate_processing_fee` | Bank transfer fee cap at $5.00; unknown payment type |

### Proposed tests to add

```python
class TestCardDeclineScenarios:
    """These simulate real gateway decline codes — critical to test."""
    def test_insufficient_funds_raises(self):
        payment = Payment(
            id="p1", order_id="o1", amount=100.00,
            payment_method=PaymentMethod(type="card", last_four="0002"),
        )
        with pytest.raises(InsufficientFundsError):
            authorize_payment(payment)

    def test_card_declined_by_issuer_raises(self):
        payment = Payment(
            id="p1", order_id="o1", amount=100.00,
            payment_method=PaymentMethod(type="card", last_four="0003"),
        )
        with pytest.raises(CardDeclinedError):
            authorize_payment(payment)

    def test_expired_card_raises(self):
        payment = Payment(
            id="p1", order_id="o1", amount=100.00,
            payment_method=PaymentMethod(
                type="card", last_four="1234",
                expiry_month=1, expiry_year=2020,
            ),
        )
        with pytest.raises(PaymentError, match="expired"):
            authorize_payment(payment)


class TestPartialRefunds:
    def test_partial_refund_sets_partially_refunded_status(self):
        payment = Payment(
            id="p1", order_id="o1", amount=100.00,
            status=PaymentStatus.CAPTURED,
        )
        result = refund_payment(payment, refund_amount=40.00)
        assert result.status == PaymentStatus.PARTIALLY_REFUNDED
        assert result.refunded_amount == 40.00

    def test_second_partial_refund_completes_refund(self):
        payment = Payment(
            id="p1", order_id="o1", amount=100.00,
            status=PaymentStatus.CAPTURED,
        )
        refund_payment(payment, 60.00)
        result = refund_payment(payment, 40.00)
        assert result.status == PaymentStatus.REFUNDED

    def test_refund_exceeds_remaining_raises(self):
        payment = Payment(
            id="p1", order_id="o1", amount=100.00,
            status=PaymentStatus.CAPTURED,
        )
        refund_payment(payment, 60.00)
        with pytest.raises(PaymentError, match="exceeds remaining"):
            refund_payment(payment, 60.00)


class TestBankTransferFeeCap:
    def test_bank_transfer_fee_capped_at_5(self):
        # 0.8% of $1000 = $8 > cap, should be $5
        fee = calculate_processing_fee(1000.00, "bank_transfer")
        assert fee == 5.00

    def test_bank_transfer_fee_below_cap(self):
        # 0.8% of $100 = $0.80 < cap
        fee = calculate_processing_fee(100.00, "bank_transfer")
        assert fee == 0.80

    def test_unknown_payment_type_returns_zero(self):
        assert calculate_processing_fee(100.00, "crypto") == 0.0
```

---

## 3. `src/notifications/service.py` — 81% — 🟠 Medium Priority

Notification failures are less catastrophic but affect user experience and
deliverability. The SMS path has **zero coverage**.

### Untested code (15 lines)

| Lines | Function | What's missing |
|-------|----------|----------------|
| 52–53 | `validate_phone_number` | Entire function — never called in tests |
| 57–59 | `truncate_sms_body` | Truncation of long SMS bodies |
| 79–87 | `send_notification` | Entire SMS channel path |
| 101, 103 | `send_notification` | PUSH channel path |
| 136 | `build_shipping_notification` | Never called in tests |

### Proposed tests to add

```python
class TestValidatePhoneNumber:
    @pytest.mark.parametrize("phone,expected", [
        ("+14155552671", True),
        ("+442071838750", True),
        ("4155552671", False),       # missing leading +
        ("+1", False),               # too short
        ("+", False),                # no digits
        ("", False),
    ])
    def test_phone_validation(self, phone, expected):
        assert validate_phone_number(phone) == expected


class TestTruncateSmsBody:
    def test_short_body_unchanged(self):
        body = "Hello!"
        assert truncate_sms_body(body) == body

    def test_exactly_160_chars_unchanged(self):
        body = "x" * 160
        assert truncate_sms_body(body) == body

    def test_long_body_truncated_with_ellipsis(self):
        body = "x" * 200
        result = truncate_sms_body(body)
        assert len(result) == 160
        assert result.endswith("...")


class TestSendSmsNotification:
    def test_send_sms_success(self):
        notif = Notification(
            id="n1", user_id="u1",
            channel=NotificationChannel.SMS,
            subject="Ship", body="Your order shipped",
            recipient="+14155552671",
        )
        result = send_notification(notif)
        assert result.status == NotificationStatus.SENT

    def test_send_sms_invalid_phone_raises(self):
        notif = Notification(
            id="n1", user_id="u1",
            channel=NotificationChannel.SMS,
            subject="Ship", body="Your order shipped",
            recipient="not-a-phone",
        )
        with pytest.raises(NotificationError):
            send_notification(notif)

    def test_long_sms_body_is_truncated_before_send(self):
        notif = Notification(
            id="n1", user_id="u1",
            channel=NotificationChannel.SMS,
            subject="Ship", body="x" * 200,
            recipient="+14155552671",
        )
        result = send_notification(notif)
        assert len(result.body) == 160


class TestRetryExceedsLimit:
    def test_retry_after_max_retries_raises(self):
        notif = Notification(
            id="n1", user_id="u1",
            channel=NotificationChannel.EMAIL,
            subject="Hi", body="body",
            recipient="user@fail.example.com",
            status=NotificationStatus.FAILED,
            retry_count=MAX_RETRIES,
        )
        with pytest.raises(NotificationError, match="maximum retries"):
            retry_failed_notification(notif)
```

---

## 4. `src/orders/models.py` — 89% — 🟡 Lower Priority

Orders has the best coverage but several error branches are still untested.

### Untested code (9 lines)

| Lines | Function | What's missing |
|-------|----------|----------------|
| 83, 85, 87, 89 | `validate_order` | Invalid quantity, negative price, bad discount, missing address |
| 99 | `apply_coupon` | Invalid coupon code raises `OrderError` |
| 114 | `cancel_order` | Cancel in shipped/delivered/refunded state raises `OrderError` |
| 128, 130 | `calculate_shipping` | `express` and `overnight` rates |
| 135 | `calculate_shipping` | Unknown shipping method raises `OrderError` |

### Proposed tests to add

```python
class TestValidateOrderErrors:
    @pytest.mark.parametrize("quantity,price,discount,addr,expected_msg", [
        (0, 10.0, 0, {"line1": "x"}, "positive quantity"),
        (1, -5.0, 0, {"line1": "x"}, "negative price"),
        (1, 10.0, 110, {"line1": "x"}, "invalid discount"),
        (1, 10.0, 0, None, "Shipping address"),
    ])
    def test_validation_errors(self, quantity, price, discount, addr, expected_msg):
        item = OrderItem("p1", "Widget", quantity, price, discount)
        order = Order(id="o1", user_id="u1", items=[item], shipping_address=addr)
        valid, errors = validate_order(order)
        assert valid is False
        assert any(expected_msg in e for e in errors)


class TestCancelInvalidStates:
    @pytest.mark.parametrize("status", [
        OrderStatus.SHIPPED,
        OrderStatus.DELIVERED,
        OrderStatus.REFUNDED,
    ])
    def test_cannot_cancel(self, status):
        order = Order(id="o1", user_id="u1", status=status)
        with pytest.raises(OrderError):
            cancel_order(order)


class TestShippingRates:
    def test_express_rate(self):
        order = Order(id="o1", user_id="u1",
                      items=[OrderItem("p1", "W", 1, 30.00)])
        assert calculate_shipping(order, "express") == 14.99

    def test_overnight_rate(self):
        order = Order(id="o1", user_id="u1",
                      items=[OrderItem("p1", "W", 1, 30.00)])
        assert calculate_shipping(order, "overnight") == 29.99

    def test_unknown_method_raises(self):
        order = Order(id="o1", user_id="u1",
                      items=[OrderItem("p1", "W", 1, 30.00)])
        with pytest.raises(OrderError, match="Unknown shipping method"):
            calculate_shipping(order, "drone")
```

---

## 5. Missing Integration Tests

There are **no integration tests at all**. The following end-to-end flows are high
value and cannot be verified by unit tests alone:

### 5a. Order placement flow

```
create user → authenticate → create order → apply coupon
  → validate order → calculate shipping → charge payment → send confirmation email
```

Critical to test:
- Coupon affects shipping fee (FREESHIP)
- Total price matches what the payment charges
- Confirmation notification contains correct order details

### 5b. Payment failure + order rollback

```
create order (CONFIRMED) → authorize payment (fails) → order stays CONFIRMED/rolls back
```

### 5c. Refund flow

```
capture payment → deliver order → request refund
  → payment becomes REFUNDED → send refund notification
```

### 5d. Account lockout flow

```
5× failed login → account locked → correct password still rejected during lockout
  → wait for lockout expiry → login succeeds
```

---

## 6. What's Entirely Untested

These areas have **0% coverage** and no tests whatsoever:

| Area | Risk | Notes |
|------|------|-------|
| `render_template` (notifications) | Medium | Template injection possible if context keys are user-controlled |
| `get_order_summary` (orders) | Low | Return shape contract could silently change |
| `get_payment_summary` (payments) | Low | Same as above |
| `build_shipping_notification` (notifications) | Medium | Phone number path uses unvalidated input |
| `require_role` (auth) | **High** | Authorization logic entirely untested |
| Integration between modules | **High** | No cross-module tests exist |

---

## 7. Recommended Action Plan

### Immediate (this sprint)

1. **Auth module** — add tests for `hash_password`, `verify_password`, session
   expiry, `revoke_session`, and `require_role`. These are security-critical paths.
2. **Payment declines** — add tests for `InsufficientFundsError`,
   `CardDeclinedError`, expired card, and partial refunds.

### Next sprint

3. **SMS notifications** — the entire SMS channel path has 0% coverage.
4. **Order validation error branches** — parametrize across all validation rules.
5. **Add integration tests** for the order-placement and refund flows.

### Ongoing

6. **Enforce a coverage gate in CI** — add `--cov-fail-under=90` to prevent
   regressions. Current baseline: 81%.
7. **Add branch coverage** — run with `--cov-branch` to catch conditional paths
   the current line-coverage metric misses (e.g., the `if/else` in
   `calculate_processing_fee` for the bank transfer cap).

---

## Coverage Command Reference

```bash
# Run all tests with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run with branch coverage (more thorough)
pytest tests/ --cov=src --cov-branch --cov-report=term-missing

# Fail CI if coverage drops below threshold
pytest tests/ --cov=src --cov-fail-under=90

# Generate HTML report for visual inspection
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```
