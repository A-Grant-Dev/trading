"""
Management Command: Train ML Models

Trains ML models (RandomForest, XGBoost) on historical market data
with full walk-forward validation and signal purging.
Now persists results to TrainingLog and saves models by default.

Usage:
    python manage.py train_models BTCUSDT
    python manage.py train_models BTCUSDT,ETHUSDT --interval 1h --limit 1000
    python manage.py train_models BTCUSDT --walk-forward --no-save
"""

import io
import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
from django.core.management.base import BaseCommand

from quant.models import TrainingLog
from quant.services.data_utils import ohlcv_to_dataframe
from quant.services.ml_features import FeaturePipeline
from quant.services.ml_models import (
    EnsemblePredictor,
    RandomForestModel,
    XGBoostModel,
)
from quant.services.ml_training import (
    ModelTrainer,
    SignalPurger,
    walk_forward_validate,
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Train ML models for trading signal generation"

    def add_arguments(self, parser):
        parser.add_argument(
            "symbols",
            type=str,
            help="Comma-separated symbols (e.g., BTCUSDT,ETHUSDT)",
        )
        parser.add_argument(
            "--interval",
            type=str,
            default="1h",
            help="Kline interval (default: 1h)",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=500,
            help="Max candles to fetch (default: 500)",
        )
        parser.add_argument(
            "--walk-forward",
            action="store_true",
            help="Run walk-forward validation",
        )
        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Don't save trained models to disk",
        )
        parser.add_argument(
            "--generate-signals",
            action="store_true",
            default=True,
            help="Generate TradeSignals from trained models (default: True)",
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
        do_walk_forward = options["walk_forward"]
        do_save = not options["no_save"]
        generate_signals = options["generate_signals"]

        pipeline = FeaturePipeline()

        for symbol in symbols:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Training models for {symbol}...")
            self.stdout.write(f"{'='*60}")

            start_time = time.time()
            log = TrainingLog.objects.create(
                model_type="ensemble",
                symbol=symbol,
                interval=interval,
                status="running",
                config={
                    "limit": limit,
                    "walk_forward": do_walk_forward,
                    "save_models": do_save,
                    "generate_signals": generate_signals,
                    "models": ["random_forest", "xgboost"],
                },
            )

            try:
                # Fetch data
                df = ohlcv_to_dataframe(symbol, interval, limit)
                if df.empty or len(df) < 100:
                    self.stderr.write(f"  Insufficient data for {symbol} ({len(df)} rows)")
                    log.status = "failed"
                    log.error_message = f"Insufficient data ({len(df)} rows)"
                    log.save()
                    continue

                self.stdout.write(f"  Data: {len(df)} {interval} candles")

                # Store symbol in attrs for feature pipeline
                df.attrs["symbol"] = symbol

                # Build features
                features = pipeline.build_features(df)
                if features.empty or len(features) < 50:
                    self.stderr.write(f"  Feature engineering failed for {symbol}")
                    log.status = "failed"
                    log.error_message = "Feature engineering produced empty result"
                    log.save()
                    continue

                self.stdout.write(f"  Features: {len(features)} rows, {len(pipeline.get_feature_names(features))} columns")

                # Prepare training data
                try:
                    X_train, X_test, y_train, y_test = pipeline.prepare_training_data(
                        features, test_size=0.2
                    )
                except ValueError as e:
                    self.stderr.write(f"  Data preparation failed: {e}")
                    log.status = "failed"
                    log.error_message = str(e)
                    log.save()
                    continue

                self.stdout.write(f"  Train: {len(X_train)} / Test: {len(X_test)}")

                # Create models
                models = []
                model_names = []

                try:
                    models.append(RandomForestModel(n_estimators=100, max_depth=10))
                    model_names.append("random_forest")
                    self.stdout.write("  ✓ RandomForest created")
                except ImportError as e:
                    self.stdout.write(f"  ✗ RandomForest unavailable: {e}")

                try:
                    models.append(XGBoostModel(n_estimators=200, max_depth=6))
                    model_names.append("xgboost")
                    self.stdout.write("  ✓ XGBoost created")
                except ImportError as e:
                    self.stdout.write(f"  ✗ XGBoost unavailable: {e}")

                if not models:
                    self.stderr.write("  No models available to train")
                    log.status = "failed"
                    log.error_message = "No ML models available (scikit-learn/xgboost not installed)"
                    log.save()
                    continue

                # Train and evaluate
                trainer = ModelTrainer()
                results = trainer.train_and_evaluate(
                    np.vstack([X_train, X_test]),
                    np.hstack([y_train, y_test]),
                    models,
                    model_names,
                )

                if "error" in results:
                    self.stderr.write(f"  Training failed: {results['error']}")
                    log.status = "failed"
                    log.error_message = results["error"]
                    log.save()
                    continue

                # Display results
                self.stdout.write(f"\n  Results:")
                feature_importance = {}
                for name, model_result in results["models"].items():
                    if model_result.get("is_trained"):
                        fi = model_result.get("feature_importance", {})
                        if fi:
                            feature_importance[name] = fi
                        self.stdout.write(
                            f"    {name}: train_acc={model_result['train_accuracy']:.1%}, "
                            f"val_acc={model_result['val_accuracy']:.1%}, "
                            f"test_acc={model_result['test_accuracy']:.1%}, "
                            f"val_sharpe={model_result['val_sharpe']:.2f}"
                        )
                    else:
                        self.stdout.write(f"    {name}: FAILED ({model_result.get('error', 'unknown')})")

                # Walk-forward validation
                wf_results = None
                if do_walk_forward and models:
                    self.stdout.write(f"\n  Walk-Forward Validation:")
                    for model, name in zip(models, model_names):
                        wf_results = walk_forward_validate(
                            model,
                            np.vstack([X_train, X_test]),
                            np.hstack([y_train, y_test]),
                        )
                        if "error" not in wf_results:
                            self.stdout.write(
                                f"    {name}: {wf_results['total_windows']} windows, "
                                f"mean_acc={wf_results['mean_accuracy']:.1%}, "
                                f"mean_sharpe={wf_results['mean_sharpe']:.2f}, "
                                f"consistency={wf_results['consistency_score']:.2f}"
                            )

                # Save models
                if do_save:
                    ensemble = EnsemblePredictor()
                    for model in models:
                        if model.is_trained:
                            ensemble.add_model(model)
                    if ensemble.models:
                        ensemble.save(symbol)
                        self.stdout.write(f"  💾 Models saved to quant/trained_models/")

                # Save training log
                duration = time.time() - start_time
                log.status = "completed"
                log.completed_at = datetime.now(timezone.utc)
                log.duration_seconds = round(duration, 2)
                log.data_points = len(features)
                log.feature_count = results.get("feature_count", 0)
                log.metrics = {
                    "models": results["models"],
                    "best_model": results["best_model"],
                    "total_samples": results["total_samples"],
                    "train_samples": results["train_samples"],
                    "val_samples": results["val_samples"],
                    "test_samples": results["test_samples"],
                    "walk_forward": wf_results,
                }
                log.feature_importance = feature_importance if feature_importance else None
                log.save()

                self.stdout.write(f"  Best model: {results['best_model']}")

                # Generate TradeSignals from trained models
                if generate_signals and models:
                    self._generate_signals(symbol, models, model_names, results)

            except Exception as e:
                logger.exception(f"Training failed for {symbol}")
                log.status = "failed"
                log.error_message = str(e)
                log.save()
                self.stderr.write(f"  ERROR: {e}")

    def _generate_signals(self, symbol: str, models: list, model_names: list, results: dict) -> None:
        """Generate TradeSignals from trained models based on their performance."""
        from quant.models import TradeSignal
        from datetime import timedelta

        # Clean up old ML ensemble signals for this symbol
        TradeSignal.objects.filter(
            symbol=symbol,
            source_model="ml_ensemble",
            status="active",
        ).update(status="expired")

        for name, model in zip(model_names, models):
            model_result = results["models"].get(name, {})
            if not model_result.get("is_trained"):
                continue

            # Only create signals if model has some predictive power
            if model_result.get("val_accuracy", 0) > 0.53:  # Better than 53% = useful
                val_sharpe = model_result.get("val_sharpe", 0)
                strength = min(1.0, max(0.3, (model_result["val_accuracy"] - 0.5) * 5))
                confidence = min(1.0, max(0.3, strength))

                # Use model to predict on latest data
                try:
                    from quant.services.data_utils import ohlcv_to_dataframe
                    df = ohlcv_to_dataframe(symbol, "1h", 100)
                    if not df.empty:
                        pipeline = FeaturePipeline()
                        features = pipeline.build_features(df)
                        if not features.empty and len(features) > 0:
                            latest = features.iloc[-1:].values.astype(float)
                            proba = model.predict_proba(latest)

                            direction = "long" if proba > 0.5 else "short"
                            TradeSignal.objects.create(
                                symbol=symbol,
                                signal_type=direction,
                                direction=direction,
                                strength=round(strength, 4),
                                confidence=round(confidence, 4),
                                source_model="ml_ensemble",
                                expiry=datetime.now(timezone.utc) + timedelta(hours=4),
                                status="active",
                                metadata={
                                    "model_name": name,
                                    "prediction": float(proba),
                                    "val_accuracy": model_result["val_accuracy"],
                                    "val_sharpe": val_sharpe,
                                    "feature_importance": model_result.get("feature_importance"),
                                },
                            )
                            self.stdout.write(f"    📡 Generated {direction} TradeSignal for {symbol} ({confidence:.0%} conf)")
                except Exception as e:
                    logger.debug(f"Signal generation failed for {name}: {e}")
