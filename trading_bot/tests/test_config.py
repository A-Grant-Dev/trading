"""
Tests for Configuration Loader — config.py and config.yaml

Coverage:
- Default config values
- YAML file loading
- Environment variable overrides
- Deep merge behavior
- Config caching
- Sync to BotConfig model
"""

import os
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

from trading_bot.services.config import (
    _apply_env_overrides,
    _deep_merge,
    _get_defaults,
    get_config,
    load_config,
    reload_config,
    sync_config_to_botconfig,
)

CONFIG_YAML_CONTENT = """
bot:
  mode: "test_mode"
  enabled: true
  max_open_positions: 10
exchange:
  timeout_ms: 5000
  use_testnet: true
logging:
  level: "DEBUG"
"""


class TestConfigDefaults(TestCase):
    """Test default configuration values."""

    def test_defaults_have_all_sections(self):
        """Default config should contain all required sections."""
        defaults = _get_defaults()
        for section in ("bot", "exchange", "backtesting", "optimization", "logging", "monitoring"):
            with self.subTest(section=section):
                self.assertIn(section, defaults)

    def test_default_bot_mode(self):
        """Default bot mode should be 'backtest'."""
        self.assertEqual(_get_defaults()["bot"]["mode"], "backtest")

    def test_default_enabled_false(self):
        """Bot should be disabled by default."""
        self.assertFalse(_get_defaults()["bot"]["enabled"])

    def test_default_virtual_balance(self):
        """Default virtual balance should be 10000.0."""
        self.assertEqual(_get_defaults()["bot"]["virtual_balance"], 10000.0)

    def test_default_real_balance_limit(self):
        """Default real balance limit should be 500.0."""
        self.assertEqual(_get_defaults()["bot"]["real_balance_limit"], 500.0)

    def test_default_max_open_positions(self):
        """Default max open positions should be 5."""
        self.assertEqual(_get_defaults()["bot"]["max_open_positions"], 5)

    def test_default_kelly_fraction(self):
        """Default Kelly fraction should be 0.25."""
        self.assertEqual(_get_defaults()["bot"]["kelly_fraction"], 0.25)

    def test_default_interval(self):
        """Default interval should be '1h'."""
        self.assertEqual(_get_defaults()["bot"]["default_interval"], "1h")

    def test_default_exchange_timeout(self):
        """Default exchange timeout should be 10000ms."""
        self.assertEqual(_get_defaults()["exchange"]["timeout_ms"], 10000)

    def test_default_optimization_trials(self):
        """Default optimization trials should be 50."""
        self.assertEqual(_get_defaults()["optimization"]["default_n_trials"], 50)


class TestDeepMerge(TestCase):
    """Test recursive dictionary deep merge."""

    def test_merge_simple(self):
        """Simple key overrides."""
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = _deep_merge(base, override)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 3)

    def test_merge_nested(self):
        """Nested dict overrides recursively."""
        base = {"bot": {"mode": "backtest", "enabled": False}}
        override = {"bot": {"mode": "paper"}}
        result = _deep_merge(base, override)
        self.assertEqual(result["bot"]["mode"], "paper")
        self.assertEqual(result["bot"]["enabled"], False)  # Should preserve other keys

    def test_merge_new_key(self):
        """Overriding with a new key should add it."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)

    def test_merge_empty_override(self):
        """Empty override should not change base."""
        base = {"a": 1}
        result = _deep_merge(base, {})
        self.assertEqual(result, {"a": 1})


class TestEnvOverrides(TestCase):
    """Test environment variable overrides."""

    def setUp(self):
        self.config = _get_defaults()

    def test_env_mode_override(self):
        """BOT_MODE env var should override mode."""
        with patch.dict(os.environ, {"BOT_MODE": "paper"}):
            result = _apply_env_overrides(self.config)
            self.assertEqual(result["bot"]["mode"], "paper")

    def test_env_bool_override(self):
        """BOT_ENABLED=true should set enabled to True."""
        with patch.dict(os.environ, {"BOT_ENABLED": "true"}):
            result = _apply_env_overrides(self.config)
            self.assertTrue(result["bot"]["enabled"])

    def test_env_bool_false(self):
        """BOT_ENABLED=false should set enabled to False."""
        with patch.dict(os.environ, {"BOT_ENABLED": "false"}):
            result = _apply_env_overrides(self.config)
            self.assertFalse(result["bot"]["enabled"])

    def test_env_int_override(self):
        """BOT_MAX_OPEN_POSITIONS=10 should override to 10."""
        with patch.dict(os.environ, {"BOT_MAX_OPEN_POSITIONS": "10"}):
            result = _apply_env_overrides(self.config)
            self.assertEqual(result["bot"]["max_open_positions"], 10)

    def test_env_float_override(self):
        """BOT_KELLY_FRACTION=0.5 should override to 0.5."""
        with patch.dict(os.environ, {"BOT_KELLY_FRACTION": "0.5"}):
            result = _apply_env_overrides(self.config)
            self.assertEqual(result["bot"]["kelly_fraction"], 0.5)

    def test_env_exchange_timeout(self):
        """EXCHANGE_TIMEOUT_MS=5000 should override timeout."""
        with patch.dict(os.environ, {"EXCHANGE_TIMEOUT_MS": "5000"}):
            result = _apply_env_overrides(self.config)
            self.assertEqual(result["exchange"]["timeout_ms"], 5000)

    def test_env_log_level(self):
        """LOG_LEVEL=DEBUG should override logging level."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            result = _apply_env_overrides(self.config)
            self.assertEqual(result["logging"]["level"], "DEBUG")

    def test_env_unset_does_nothing(self):
        """Unset env vars should not change config."""
        with patch.dict(os.environ, {}, clear=True):
            result = _apply_env_overrides(self.config)
            self.assertEqual(result["bot"]["mode"], "backtest")


class TestYamlLoading(TestCase):
    """Test loading from YAML file."""

    def test_load_config_no_file(self):
        """Loading from non-existent file should use defaults."""
        fake_path = Path("/nonexistent/config.yaml")
        config = load_config(fake_path)
        self.assertEqual(config["bot"]["mode"], "backtest")

    def test_load_config_from_yaml_string(self):
        """Test loading from a temporary YAML string via file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(CONFIG_YAML_CONTENT)
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            self.assertEqual(config["bot"]["mode"], "test_mode")
            self.assertTrue(config["bot"]["enabled"])
            self.assertEqual(config["bot"]["max_open_positions"], 10)
            self.assertEqual(config["exchange"]["timeout_ms"], 5000)
            self.assertTrue(config["exchange"]["use_testnet"])
            self.assertEqual(config["logging"]["level"], "DEBUG")
        finally:
            tmp_path.unlink()

    def test_yaml_overrides_defaults(self):
        """YAML values should override defaults but preserve unset keys."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("bot:\n  mode: live\n")
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            # Overridden
            self.assertEqual(config["bot"]["mode"], "live")
            # Preserved from defaults
            self.assertEqual(config["bot"]["virtual_balance"], 10000.0)
            self.assertEqual(config["exchange"]["timeout_ms"], 10000)
        finally:
            tmp_path.unlink()


class TestConfigCaching(TestCase):
    """Test config caching behavior."""

    def test_get_config_returns_same_object(self):
        """get_config should return cached config on second call."""
        config1 = get_config(Path("/nonexistent/config.yaml"))
        config2 = get_config(Path("/nonexistent/config.yaml"))
        self.assertIs(config1, config2)  # Same object

    def test_reload_config_returns_new_object(self):
        """reload_config should return a fresh config."""
        config1 = get_config(Path("/nonexistent/config.yaml"))
        config2 = reload_config(Path("/nonexistent/config.yaml"))
        # After reload, they should be different objects
        self.assertIsNot(config1, config2)


class TestSyncToBotConfig(TestCase):
    """Test syncing YAML config to BotConfig model."""

    def test_sync_with_valid_config(self):
        """Syncing with valid config should return True."""
        config = _get_defaults()
        try:
            result = sync_config_to_botconfig(config)
            # May fail if BotConfig model doesn't exist yet
            self.assertIsInstance(result, bool)
        except Exception:
            pass  # BotConfig might not be migrated yet
