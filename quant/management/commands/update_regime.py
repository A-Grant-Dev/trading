"""
Management Command: Update Market Regime

Periodically fetches the latest market data, rebuilds HMM features,
and recomputes the current market regime for each tracked symbol.
Now persists results to TrainingLog and generates TradeSignals.

Usage:
    python manage.py update_regime
    python manage.py update_regime --symbols BTCUSDT,ETHUSDT
    python manage.py update_regime --interval 5m --limit 500
    python manage.py update_regime --generate-signals

Renaissance principle: The regime must be continuously updated.
Markets change character without warning, and the model must adapt.
"""

import logging
import os
import pickle
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from quant.models import TradeSignal, TrainingLog
from quant.services.data_utils import ohlcv_to_dataframe
from quant.services.hmm_regime import (
    MarketRegimeDetector,
    build_hmm_features,
    DEFAULT_FEATURES,
)
from quant.services.data_feeds import get_cache, set_cache

logger = logging.getLogger(__name__)

# Default symbols to track for regime detection
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# ── Model Persistence ──────────────────────────────────────────────

MODEL_DIR = Path(settings.BASE_DIR) / "quant" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(symbol: str) -> Path:
    """Get the file path for a saved HMM model."""
    return MODEL_DIR / f"hmm_{symbol.lower()}.pkl"


def _save_detector(symbol: str, detector: MarketRegimeDetector) -> None:
    """Persist a trained detector to disk."""
    path = _model_path(symbol)
    try:
        with open(path, "wb") as f:
            pickle.dump(detector, f)
        logger.info(f"Saved HMM model for {symbol} to {path}")
    except Exception as e:
        logger.error(f"Failed to save HMM model for {symbol}: {e}")


def _load_detector(symbol: str) -> MarketRegimeDetector | None:
    """Load a previously trained detector from disk."""
    path = _model_path(symbol)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            detector = pickle.load(f)
        logger.info(f"Loaded HMM model for {symbol} from {path}")
        return detector
    except Exception as e:
        logger.warning(f"Failed to load HMM model for {symbol}: {e}")
        return None


class Command(BaseCommand):
    help = "Update HMM market regime for tracked symbols"

    def add_arguments(self, parser):
        parser.add_argument(
            "--symbols",
            type=str,
            default=",".join(DEFAULT_SYMBOLS),
            help=f"Comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            help="Kline interval for regime detection (default: 1h)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=500,
            help="Number of candles to fetch (default: 500)",
        )
        parser.add_argument(
            "--force-retrain",
            action="store_true",
            help="Force retrain the HMM model even if already trained",
        )
        parser.add_argument(
            "--generate-signals",
            action="store_true",
            default=True,
            help="Generate TradeSignals from regime data (default: True)",
        )
        parser.add_argument(
            "--no-signals",
            action="store_false",
            dest="generate_signals",
            help="Skip TradeSignal generation",
        )

    def handle(self, *args, **options):
        symbols = [s.strip().upper() for s in options["symbols"].split(",")]
        interval = options["interval"]
        limit = options["limit"]
        force_retrain = options["force_retrain"]
        generate_signals = options["generate_signals"]

        self.stdout.write(f"Updating regime for {len(symbols)} symbols [{', '.join(symbols)}]...")

        results = {}
        for symbol in symbols:
            start_time = time.time()
            log = TrainingLog.objects.create(
                model_type="hmm_regime",
                symbol=symbol,
                interval=interval,
                status="running",
                config={
                    "limit": limit,
                    "force_retrain": force_retrain,
                    "n_states": 4,
                },
            )
            try:
                result = self._update_symbol_regime(symbol, interval, limit, force_retrain)
                result["symbol"] = symbol
                results[symbol] = result

                # Save training log
                duration = time.time() - start_time
                log.status = "completed"
                log.completed_at = datetime.now(timezone.utc)
                log.duration_seconds = round(duration, 2)
                log.data_points = result.get("data_points", 0)
                log.metrics = {
                    "regime_label": result.get("regime_label", "unknown"),
                    "regime_id": result.get("regime_id", -1),
                    "top_probability": result.get("top_probability", 0),
                    "probabilities": result.get("probabilities", {}),
                    "n_states": 4,
                }
                log.save()

                self.stdout.write(
                    f"  {symbol}: {result['regime_label']} "
                    f"(conf: {result['top_probability']:.1%}) "
                    f"[{duration:.1f}s]"
                )

                # Generate TradeSignals from regime data
                if generate_signals and result["regime_label"] != "unknown":
                    self._generate_signal(symbol, result)

            except Exception as e:
                logger.exception(f"Failed to update regime for {symbol}")
                log.status = "failed"
                log.error_message = str(e)
                log.completed_at = datetime.now(timezone.utc)
                log.save()
                results[symbol] = {"error": str(e)}
                self.stderr.write(f"  {symbol}: ERROR — {e}")

        # Cache the full results
        set_cache("regime:all", results)

        self.stdout.write(self.style.SUCCESS(f"Regime update complete for {len(symbols)} symbols"))

    def _generate_signal(self, symbol: str, result: dict) -> None:
        """
        Generate TradeSignals from regime detection results.
        
        Creates signals when regime indicates trading opportunities:
          - bullish → long signal
          - bearish → short signal
          - ranging → neutral (no direction)
          - volatile → no signal (too risky)
        """
        label = result["regime_label"]
        confidence = result["top_probability"]

        # Deactivate old regime signals for this symbol
        TradeSignal.objects.filter(
            symbol=symbol,
            source_model="hmm_regime",
            status="active",
        ).update(status="expired")

        # Only create direction signals for trending regimes
        direction = None
        if label == "bullish" and confidence > 0.5:
            direction = "long"
        elif label == "bearish" and confidence > 0.5:
            direction = "short"

        if direction:
            TradeSignal.objects.create(
                symbol=symbol,
                signal_type=direction,
                direction=direction,
                strength=round(confidence, 4),
                confidence=round(confidence, 4),
                source_model="hmm_regime",
                expiry=datetime.now(timezone.utc) + timedelta(hours=6),
                status="active",
                metadata={
                    "regime": label,
                    "regime_id": result["regime_id"],
                    "top_probability": confidence,
                    "probabilities": result.get("probabilities", {}),
                    "signal_type": "regime_based",
                },
            )

    def _update_symbol_regime(
        self,
        symbol: str,
        interval: str,
        limit: int,
        force_retrain: bool,
    ) -> dict:
        """Update regime for a single symbol."""
        # Fetch data
        df = ohlcv_to_dataframe(symbol, interval, limit)
        if df.empty or len(df) < 50:
            return {
                "symbol": symbol,
                "regime_label": "unknown",
                "regime_id": -1,
                "top_probability": 0.0,
                "error": "Insufficient data",
            }

        # Build HMM features
        features = build_hmm_features(df)

        if features.empty or len(features) < 30:
            return {
                "symbol": symbol,
                "regime_label": "unknown",
                "regime_id": -1,
                "top_probability": 0.0,
                "error": "Insufficient features after processing",
            }

        # Get or create detector (load from disk cache first, then retrain if needed)
        detector = None if force_retrain else _load_detector(symbol)
        if detector is None or not detector.is_trained:
            detector = MarketRegimeDetector(n_states=4)
            try:
                detector.train(features, DEFAULT_FEATURES)
                _save_detector(symbol, detector)
            except (ValueError, Exception) as e:
                return {
                    "symbol": symbol,
                    "regime_label": "unknown",
                    "regime_id": -1,
                    "top_probability": 0.0,
                    "error": f"Training failed: {e}",
                }

        # Predict current regime
        latest_features = features[DEFAULT_FEATURES].iloc[-1:].values.flatten()
        regime_id, regime_label = detector.predict_regime(latest_features)

        # Get probabilities
        probabilities = detector.get_regime_probabilities(latest_features)
        top_prob = list(probabilities.values())[0]["probability"] if probabilities else 0.0

        # Build result
        result = {
            "symbol": symbol,
            "interval": interval,
            "regime_id": regime_id,
            "regime_label": regime_label,
            "top_probability": top_prob,
            "probabilities": probabilities,
            "regime_info": detector.get_regime_info(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "data_points": len(features),
            "is_trained": detector.is_trained,
        }

        # Store in cache
        set_cache(f"regime:{symbol}", result)

        return result
