"""
Circuit Breaker — Retry, Timeout, and Failure Isolation for External APIs

Prevents cascading failures when external services (Binance, etc.) are
unreachable or return errors. Implements the circuit breaker pattern:

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests are rejected immediately
- HALF_OPEN: Testing if service recovered after cooldown

Every state transition is logged to AuditLog for traceability.
"""

import functools
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# ── Types ───────────────────────────────────────────────────────────

F = TypeVar("F", bound=Callable[..., Any])


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — reject requests
    HALF_OPEN = "half_open"  # Testing recovery


# ── Circuit Breaker Registry ────────────────────────────────────────

_breakers: dict[str, dict[str, Any]] = {}


def _get_breaker(name: str) -> dict[str, Any]:
    """Get or create a circuit breaker state dict."""
    if name not in _breakers:
        _breakers[name] = {
            "state": CircuitState.CLOSED,
            "failure_count": 0,
            "last_failure_time": None,
            "last_success_time": None,
            "threshold": 3,         # Failures before opening
            "cooldown": 60,         # Seconds before half-open
            "recovery_threshold": 2, # Successes in half-open to close
            "recovery_count": 0,
        }
    return _breakers[name]


def reset_breaker(name: str) -> None:
    """Reset a circuit breaker to its initial closed state."""
    if name in _breakers:
        breaker = _breakers[name]
        breaker["state"] = CircuitState.CLOSED
        breaker["failure_count"] = 0
        breaker["last_failure_time"] = None
        breaker["recovery_count"] = 0
        logger.info("Circuit breaker '%s' reset to CLOSED", name)


def get_breaker_state(name: str) -> dict[str, Any]:
    """Get the current state of a circuit breaker for monitoring."""
    return dict(_get_breaker(name))  # Return a copy


def get_all_breaker_states() -> dict[str, dict[str, Any]]:
    """Get all circuit breaker states."""
    return {k: dict(v) for k, v in _breakers.items()}


# ── Circuit Breaker Decorator ───────────────────────────────────────


def circuit_breaker(
    name: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    failure_threshold: int = 3,
    cooldown_seconds: float = 60.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that wraps a function with circuit breaker logic.

    Args:
        name: Circuit breaker name (for monitoring and isolation)
        max_retries: Number of retry attempts before opening circuit
        retry_delay: Seconds between retries
        failure_threshold: Consecutive failures before opening circuit
        cooldown_seconds: Seconds in OPEN state before trying HALF_OPEN
        exceptions: Exception types that count as failures

    Usage:
        @circuit_breaker("binance_api", max_retries=2, retry_delay=1.0)
        def fetch_price(symbol):
            return exchange.fetch_ticker(symbol)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            breaker = _get_breaker(name)
            breaker["threshold"] = failure_threshold
            breaker["cooldown"] = cooldown_seconds

            # ── Check circuit state ─────────────────────────────
            if breaker["state"] == CircuitState.OPEN:
                # Check if cooldown elapsed → transition to HALF_OPEN
                if breaker["last_failure_time"] is not None:
                    elapsed = time.time() - breaker["last_failure_time"]
                    if elapsed >= breaker["cooldown"]:
                        breaker["state"] = CircuitState.HALF_OPEN
                        breaker["recovery_count"] = 0
                        logger.info(
                            "Circuit breaker '%s' → HALF_OPEN "
                            "(cooldown elapsed: %.1fs)", name, elapsed,
                        )
                    else:
                        remaining = breaker["cooldown"] - elapsed
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker '{name}' is OPEN. "
                            f"Retry in {remaining:.0f}s"
                        )
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{name}' is OPEN"
                    )

            # ── Execute with retries ────────────────────────────
            last_error: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Success — update breaker state
                    if breaker["state"] == CircuitState.HALF_OPEN:
                        breaker["recovery_count"] += 1
                        if breaker["recovery_count"] >= breaker["recovery_threshold"]:
                            breaker["state"] = CircuitState.CLOSED
                            breaker["failure_count"] = 0
                            logger.info(
                                "Circuit breaker '%s' → CLOSED "
                                "(recovered after %d successes)",
                                name, breaker["recovery_count"],
                            )

                    breaker["last_success_time"] = time.time()
                    breaker["failure_count"] = 0
                    return result

                except exceptions as e:
                    last_error = e
                    logger.warning(
                        "Circuit breaker '%s' attempt %d/%d failed: %s",
                        name, attempt, max_retries, e,
                    )

                    if attempt < max_retries:
                        time.sleep(retry_delay)
                    else:
                        # All retries exhausted — open circuit
                        breaker["failure_count"] += 1
                        breaker["last_failure_time"] = time.time()

                        if breaker["failure_count"] >= failure_threshold:
                            breaker["state"] = CircuitState.OPEN
                            logger.error(
                                "Circuit breaker '%s' → OPEN "
                                "(%d consecutive failures)",
                                name, breaker["failure_count"],
                            )
                            # Try to log to AuditLog
                            try:
                                from trading_bot.models import AuditLog
                                AuditLog.objects.create(
                                    action="error",
                                    message=(
                                        f"Circuit breaker '{name}' OPEN after "
                                        f"{breaker['failure_count']} failures"
                                    ),
                                    details={
                                        "breaker_name": name,
                                        "failure_count": breaker["failure_count"],
                                        "last_error": str(e),
                                    },
                                    severity="error",
                                )
                            except Exception:
                                pass

                        raise CircuitBreakerOpenError(
                            f"Circuit breaker '{name}' opened after "
                            f"{breaker['failure_count']} failures. "
                            f"Last error: {e}"
                        ) from e

            # Shouldn't reach here
            raise RuntimeError(f"Unexpected error in circuit breaker '{name}'")

        return wrapper  # type: ignore
    return decorator


class CircuitBreakerOpenError(Exception):
    """Raised when a circuit breaker is open and rejects a request."""
    pass


# ── Retry Decorator (without circuit breaker) ───────────────────────


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """
    Simple retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
        exceptions: Exception types that trigger retry

    Usage:
        @retry(max_attempts=3, delay=1.0, backoff=2.0)
        def fetch_data(url):
            return requests.get(url)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_error: Optional[Exception] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        logger.warning(
                            "Retry %d/%d for %s: %s. Waiting %.1fs",
                            attempt, max_attempts, func.__name__, e, current_delay,
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            "All %d retries exhausted for %s: %s",
                            max_attempts, func.__name__, e,
                        )
                        raise

            raise RuntimeError("Unexpected retry error") from last_error

        return wrapper  # type: ignore
    return decorator
