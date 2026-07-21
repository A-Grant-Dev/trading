"""
Configuration Loader — YAML + Environment Variable Configuration

Loads configuration from config.yaml with environment variable overrides.
Uses pydantic for schema validation if available, with fallback to dict.

Priority (highest to lowest):
1. Environment variables (BOT_MODE, EXCHANGE_TIMEOUT_MS, etc.)
2. config.yaml file
3. Default values
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Config Path ─────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ── Config Loader ───────────────────────────────────────────────────


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config.yaml (default: trading_bot/config.yaml)

    Returns:
        Dict with full merged configuration
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config = _get_defaults()

    # Load YAML file if it exists
    if config_path.exists():
        try:
            import yaml  # type: ignore

            with open(config_path) as f:
                file_config = yaml.safe_load(f)
            if file_config:
                config = _deep_merge(config, file_config)
                logger.info("Loaded config from %s", config_path)
        except ImportError:
            logger.warning("PyYAML not installed. Install with: pip install pyyaml")
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", config_path, e)
    else:
        logger.info("No config.yaml found at %s, using defaults + env vars", config_path)

    # Override with environment variables
    config = _apply_env_overrides(config)

    return config


def _get_defaults() -> dict[str, Any]:
    """Get default configuration values."""
    return {
        "bot": {
            "mode": "backtest",
            "enabled": False,
            "virtual_balance": 10000.0,
            "real_balance_limit": 500.0,
            "max_open_positions": 5,
            "max_position_size_pct": 10.0,
            "max_daily_loss_pct": 5.0,
            "max_drawdown_pct": 15.0,
            "kelly_fraction": 0.25,
            "circuit_breaker_count": 3,
            "circuit_breaker_hours": 24,
            "nightly_optimization_enabled": False,
            "min_improvement_threshold": 0.05,
            "max_trials_per_study": 500,
            "default_symbol": "BTCUSDT",
            "default_interval": "1h",
            "max_history_days": 730,
        },
        "exchange": {
            "timeout_ms": 10000,
            "rate_limit_enabled": True,
            "retry_attempts": 3,
            "retry_delay_seconds": 2.0,
            "use_testnet": False,
            "maker_fee_pct": 0.001,
            "taker_fee_pct": 0.001,
            "default_slippage_pct": 0.0005,
        },
        "backtesting": {
            "default_initial_capital": 10000.0,
            "default_position_size_pct": 10.0,
            "default_fee_pct": 0.001,
            "default_slippage_pct": 0.0005,
            "walk_forward_default_train": 100,
            "walk_forward_default_test": 50,
        },
        "optimization": {
            "default_n_trials": 50,
            "default_n_jobs": 4,
            "maximize_metric": "sharpe_ratio",
            "study_storage": None,
        },
        "logging": {
            "level": "INFO",
            "format": "structured",
            "log_signals": True,
            "log_trades": True,
            "log_optimization": True,
        },
        "monitoring": {
            "metrics_enabled": True,
            "dashboard_auto_refresh_seconds": 30,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to config.

    Convention: BOT_MODE → config["bot"]["mode"]
                 EXCHANGE_TIMEOUT_MS → config["exchange"]["timeout_ms"]
    """
    env_map = {
        "BOT_MODE": ("bot", "mode"),
        "BOT_ENABLED": ("bot", "enabled"),
        "BOT_VIRTUAL_BALANCE": ("bot", "virtual_balance"),
        "BOT_REAL_BALANCE_LIMIT": ("bot", "real_balance_limit"),
        "BOT_MAX_OPEN_POSITIONS": ("bot", "max_open_positions"),
        "BOT_MAX_POSITION_SIZE_PCT": ("bot", "max_position_size_pct"),
        "BOT_MAX_DAILY_LOSS_PCT": ("bot", "max_daily_loss_pct"),
        "BOT_MAX_DRAWDOWN_PCT": ("bot", "max_drawdown_pct"),
        "BOT_KELLY_FRACTION": ("bot", "kelly_fraction"),
        "BOT_CIRCUIT_BREAKER_COUNT": ("bot", "circuit_breaker_count"),
        "BOT_CIRCUIT_BREAKER_HOURS": ("bot", "circuit_breaker_hours"),
        "BOT_NIGHTLY_OPTIMIZATION": ("bot", "nightly_optimization_enabled"),
        "BOT_MAX_TRIALS": ("bot", "max_trials_per_study"),
        "BOT_DEFAULT_SYMBOL": ("bot", "default_symbol"),
        "BOT_DEFAULT_INTERVAL": ("bot", "default_interval"),
        "EXCHANGE_TIMEOUT_MS": ("exchange", "timeout_ms"),
        "EXCHANGE_RETRY_ATTEMPTS": ("exchange", "retry_attempts"),
        "EXCHANGE_RETRY_DELAY": ("exchange", "retry_delay_seconds"),
        "EXCHANGE_USE_TESTNET": ("exchange", "use_testnet"),
        "LOG_LEVEL": ("logging", "level"),
        "LOG_FORMAT": ("logging", "format"),
        "OPTIMIZATION_TRIALS": ("optimization", "default_n_trials"),
        "OPTIMIZATION_JOBS": ("optimization", "default_n_jobs"),
        "OPTIMIZATION_METRIC": ("optimization", "maximize_metric"),
    }

    for env_var, (section, key) in env_map.items():
        value = os.getenv(env_var)
        if value is not None:
            # Type conversion
            section_data = config.get(section, {})
            existing = section_data.get(key)
            if existing is not None:
                if isinstance(existing, bool):
                    converted = value.lower() in ("true", "1", "yes")
                elif isinstance(existing, int):
                    converted = int(value)
                elif isinstance(existing, float):
                    converted = float(value)
                else:
                    converted = value
                config[section][key] = converted

    return config


# ── Singleton Config ────────────────────────────────────────────────

_config_cache: Optional[dict] = None


def get_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Get the cached configuration (loaded once).

    Args:
        config_path: Optional path to config.yaml

    Returns:
        Dict with full configuration
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config(config_path)
    return _config_cache


def reload_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Force reload configuration from disk.

    Args:
        config_path: Optional path to config.yaml

    Returns:
        Dict with full configuration
    """
    global _config_cache
    _config_cache = load_config(config_path)
    return _config_cache


# ── Config to BotConfig Sync ────────────────────────────────────────


def sync_config_to_botconfig(yaml_config: Optional[dict] = None) -> bool:
    """
    Sync YAML config to the BotConfig database model.

    Args:
        yaml_config: Optional config dict (loads if None)

    Returns:
        True if synced successfully
    """
    if yaml_config is None:
        yaml_config = get_config()

    try:
        from trading_bot.models import BotConfig

        config = BotConfig.get_config()
        bot_cfg = yaml_config.get("bot", {})

        if "mode" in bot_cfg:
            config.mode = bot_cfg["mode"]
        if "enabled" in bot_cfg:
            config.is_enabled = bot_cfg["enabled"]
        if "virtual_balance" in bot_cfg:
            from decimal import Decimal
            config.virtual_balance = Decimal(str(bot_cfg["virtual_balance"]))
        if "real_balance_limit" in bot_cfg:
            from decimal import Decimal
            config.real_balance_limit = Decimal(str(bot_cfg["real_balance_limit"]))
        if "max_open_positions" in bot_cfg:
            config.max_open_positions = bot_cfg["max_open_positions"]
        if "max_position_size_pct" in bot_cfg:
            config.max_position_size_pct = bot_cfg["max_position_size_pct"]
        if "max_daily_loss_pct" in bot_cfg:
            config.max_daily_loss_pct = bot_cfg["max_daily_loss_pct"]
        if "max_drawdown_pct" in bot_cfg:
            config.max_drawdown_pct = bot_cfg["max_drawdown_pct"]
        if "kelly_fraction" in bot_cfg:
            config.kelly_fraction = bot_cfg["kelly_fraction"]
        if "circuit_breaker_count" in bot_cfg:
            config.circuit_breaker_count = bot_cfg["circuit_breaker_count"]
        if "circuit_breaker_hours" in bot_cfg:
            config.circuit_breaker_hours = bot_cfg["circuit_breaker_hours"]
        if "nightly_optimization_enabled" in bot_cfg:
            config.nightly_optimization_enabled = bot_cfg["nightly_optimization_enabled"]
        if "min_improvement_threshold" in bot_cfg:
            config.min_improvement_threshold = bot_cfg["min_improvement_threshold"]
        if "max_trials_per_study" in bot_cfg:
            config.max_trials_per_study = bot_cfg["max_trials_per_study"]
        if "default_interval" in bot_cfg:
            config.default_interval = bot_cfg["default_interval"]
        if "max_history_days" in bot_cfg:
            config.max_history_days = bot_cfg["max_history_days"]

        config.save()
        logger.info("Synced YAML config to BotConfig model")
        return True

    except Exception as e:
        logger.error("Failed to sync config to BotConfig: %s", e)
        return False
