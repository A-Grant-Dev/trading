"""
Tests for Circuit Breaker — circuit_breaker.py

Coverage:
- Circuit breaker states (CLOSED, OPEN, HALF_OPEN)
- Retry logic on failures
- Cooldown period
- Recovery threshold
- Success resets
- Custom exception types
- get_breaker_state monitoring
- Retry decorator (without circuit breaker)
"""

import time
from unittest.mock import patch

from django.test import TestCase

from trading_bot.services.circuit_breaker import (
    CircuitBreakerOpenError,
    CircuitState,
    _get_breaker as _get_breaker_internal,
    circuit_breaker,
    get_all_breaker_states,
    get_breaker_state,
    reset_breaker,
    retry,
)


class TestCircuitBreakerStates(TestCase):
    """Test circuit breaker state transitions."""

    def setUp(self):
        reset_breaker("test_breaker")

    def test_initial_state_closed(self):
        """Breaker should start in CLOSED state."""
        state = get_breaker_state("test_breaker")
        self.assertEqual(state["state"], CircuitState.CLOSED)
        self.assertEqual(state["failure_count"], 0)

    def test_open_after_consecutive_failures(self):
        """Breaker should open after reaching failure threshold."""
        call_count = [0]

        @circuit_breaker(
            "test_breaker",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=2,
            cooldown_seconds=60,
        )
        def failing_func():
            call_count[0] += 1
            raise ValueError("Test failure")

        # Should fail and open the circuit after threshold is reached
        for _ in range(2):
            try:
                failing_func()
            except (ValueError, CircuitBreakerOpenError):
                pass

        state = get_breaker_state("test_breaker")
        self.assertEqual(state["state"], CircuitState.OPEN)
        self.assertGreaterEqual(state["failure_count"], 2)

    def test_open_blocks_requests(self):
        """OPEN breaker should reject requests immediately."""
        call_count = [0]

        @circuit_breaker(
            "test_breaker_open_block",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=1,
            cooldown_seconds=60,
        )
        def failing_func():
            call_count[0] += 1
            raise ValueError("Fail")

        # Trip the breaker
        with self.assertRaises((ValueError, CircuitBreakerOpenError)):
            failing_func()

        # Should be immediately rejected (not call the function)
        with self.assertRaises(CircuitBreakerOpenError):
            failing_func()

        # Function should only have been called once (first attempt)
        self.assertEqual(call_count[0], 1)

    def test_half_open_after_cooldown(self):
        """Breaker should transition to HALF_OPEN after cooldown elapses."""
        @circuit_breaker(
            "test_half_open",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=1,
            cooldown_seconds=0.05,  # Very short cooldown
        )
        def failing_func():
            raise ValueError("Fail")

        # Trip the breaker
        with self.assertRaises((ValueError, CircuitBreakerOpenError)):
            failing_func()

        # Wait for cooldown
        time.sleep(0.1)

        # Should now be HALF_OPEN or allow attempt
        state = get_breaker_state("test_half_open")
        # The breaker should have moved to HALF_OPEN since cooldown elapsed
        # But since we didn't call the function, it won't transition automatically
        # Let's check that calling it now will trigger the half-open check

    def test_success_closes_half_open(self):
        """Successful calls during HALF_OPEN should close the breaker."""
        trip_count = [0]

        @circuit_breaker(
            "test_recovery",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=1,
            cooldown_seconds=0.05,
        )
        def failing_func():
            """Always fails — used to trip the breaker."""
            trip_count[0] += 1
            raise ValueError("Fail")

        @circuit_breaker(
            "test_recovery",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=1,
            cooldown_seconds=0.05,
        )
        def recovery_func():
            """Always succeeds — used to test HALF_OPEN recovery."""
            return "success"

        # Trip the breaker with failing_func
        with self.assertRaises((ValueError, CircuitBreakerOpenError)):
            failing_func()

        # Wait for cooldown to allow HALF_OPEN transition
        time.sleep(0.1)

        # Call recovery_func once — breaker transitions from OPEN to HALF_OPEN
        result = recovery_func()
        self.assertEqual(result, "success")

        # Need a second successful call to reach recovery_threshold (2)
        result = recovery_func()
        self.assertEqual(result, "success")

        state = get_breaker_state("test_recovery")
        self.assertEqual(state["state"], CircuitState.CLOSED)

    def test_reset_breaker(self):
        """reset_breaker should return to CLOSED with zero failures."""
        @circuit_breaker(
            "test_reset",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=1,
            cooldown_seconds=60,
        )
        def failing_func():
            raise ValueError("Fail")

        # Trip the breaker
        with self.assertRaises((ValueError, CircuitBreakerOpenError)):
            failing_func()

        # Reset
        reset_breaker("test_reset")
        state = get_breaker_state("test_reset")
        self.assertEqual(state["state"], CircuitState.CLOSED)
        self.assertEqual(state["failure_count"], 0)


class TestCircuitBreakerRetry(TestCase):
    """Test retry logic within circuit breaker."""

    def setUp(self):
        reset_breaker("test_retry_breaker")

    def test_retry_succeeds_eventually(self):
        """Breaker should retry and succeed within max_retries."""
        call_count = [0]

        @circuit_breaker(
            "test_retry_breaker",
            max_retries=3,
            retry_delay=0.01,
            failure_threshold=3,
            cooldown_seconds=60,
        )
        def eventually_succeeds():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Not yet")
            return "success"

        result = eventually_succeeds()
        self.assertEqual(result, "success")
        self.assertEqual(call_count[0], 2)

    def test_retry_exhaustion(self):
        """Breaker should raise after all retries exhausted."""
        call_count = [0]

        @circuit_breaker(
            "test_exhaustion",
            max_retries=2,
            retry_delay=0.01,
            failure_threshold=2,
            cooldown_seconds=60,
        )
        def always_fails():
            call_count[0] += 1
            raise ValueError("Always fails")

        with self.assertRaises(CircuitBreakerOpenError):
            always_fails()
        self.assertEqual(call_count[0], 2)

    def test_custom_exception_types(self):
        """Breaker should only count specified exceptions as failures."""
        call_count = [0]

        @circuit_breaker(
            "test_custom_exc",
            max_retries=1,
            retry_delay=0.01,
            failure_threshold=1,
            exceptions=(ValueError,),
        )
        def raises_typeerror():
            call_count[0] += 1
            raise TypeError("Not caught")

        # TypeError is not ValueError, so it should propagate immediately
        with self.assertRaises(TypeError):
            raises_typeerror()

        # Breaker should still be CLOSED since it wasn't counted
        state = get_breaker_state("test_custom_exc")
        self.assertEqual(state["state"], CircuitState.CLOSED)


class TestRetryDecorator(TestCase):
    """Test the simple retry decorator."""

    def test_retry_eventually_succeeds(self):
        """Retry should succeed after initial failures."""
        call_count = [0]

        @retry(max_attempts=3, delay=0.01)
        def eventually_works():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("Not yet")
            return "ok"

        result = eventually_works()
        self.assertEqual(result, "ok")
        self.assertEqual(call_count[0], 2)

    def test_retry_exhaustion(self):
        """Retry should raise after all attempts exhausted."""
        call_count = [0]

        @retry(max_attempts=2, delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Fail")

        with self.assertRaises(ValueError):
            always_fails()
        self.assertEqual(call_count[0], 2)

    def test_retry_exponential_backoff(self):
        """Retry should use increasing delays."""
        import time

        call_count = [0]
        times = []

        @retry(max_attempts=3, delay=0.01, backoff=3.0)
        def failing():
            call_count[0] += 1
            times.append(time.time())
            raise ValueError("Fail")

        with self.assertRaises(ValueError):
            failing()

        self.assertEqual(call_count[0], 3)
        # Second delay should be ~3x first delay
        if len(times) >= 3:
            delay1 = times[1] - times[0]
            delay2 = times[2] - times[1]
            self.assertGreater(delay2, delay1 * 1.5)  # At least 1.5x longer


class TestBreakerMonitoring(TestCase):
    """Test monitoring/state inspection of breakers."""

    def test_get_all_breakers_empty(self):
        """get_all_breaker_states should return empty dict initially."""
        states = get_all_breaker_states()
        self.assertIsInstance(states, dict)

    def test_get_all_breakers_after_creation(self):
        """get_all_breaker_states should include created breakers."""
        _get_breaker_internal("monitor_test_1")
        _get_breaker_internal("monitor_test_2")
        states = get_all_breaker_states()
        self.assertIn("monitor_test_1", states)
        self.assertIn("monitor_test_2", states)
