# Autonomous Spot Trading Bot – Complete Development Plan
**Target:** Django project (already exists + Binance connected via ccxt)  
**Goal:** Self-improving spot trading system that uses every data source and strategy listed below, runs massive backtests, and continuously optimizes parameters for highest possible win rate.

This document is written specifically for a coding agent. Follow phases strictly. Do not skip or invent features outside this scope. All code must be production-grade, fully typed, logged, and idempotent.

---

## 1. Core Requirements (Non-Negotiable)

- New Django app: `trading_bot`
- Must use **all** data sources listed in Section 2
- Must implement **all** interpretation strategies in Section 3
- Must implement the full self-improvement stack in Section 4
- Vectorized backtesting + walk-forward + combinatorial purged CV
- Optuna + Ray hyperparameter search
- Paper trading → Live promotion flow
- Nightly automated retrain + promote best ParamSet
- Zero hard-coded secrets, full audit trail, circuit breakers

---

## 2. Data Sources (Implement Every Single One)

### Market Data
- Multi-exchange OHLCV (1s, 5s, 15s, 1m, 5m, 15m, 1h, 4h, 1d) – start with Binance, architecture must support others
- Full L2/L3 order book depth + real-time updates
- Tick-by-tick trades + aggregated volume profile
- Real-time funding rates, open interest, liquidations (from USDT-M futures even for spot decisions)

### On-Chain
- Exchange net flows (in/out)
- Whale wallet movements (top 100–1000)
- Active addresses, new addresses
- Realized profit/loss, SOPR, MVRV, MVRV Z-Score, NVT
- Entity-adjusted metrics
- Sources: free Glassnode endpoints, CryptoQuant free tier, Dune Analytics SQL, web3.py direct RPC

### Derivatives & Options
- Funding rate extremes
- Open interest delta
- Liquidation cascades
- Options IV surface, 25-delta skew, put/call ratio (Deribit or Binance options)

### Cross-Asset & Macro
- BTC Dominance, ETH/BTC ratio
- DXY, SPX, Gold, US10Y, Fed funds futures
- Economic calendar + surprise index

### Sentiment & Alternative
- X (Twitter) volume + NLP sentiment embeddings
- Reddit (r/cryptocurrency, r/bitcoin) volume + sentiment
- Fear & Greed Index
- News headline embeddings (CryptoPanic / RSS)
- Google Trends (Bitcoin, crypto)
- Stablecoin supply changes (USDT, USDC)
- GitHub development activity (Bitcoin, Ethereum)

### Microstructure
- Bid-ask spread
- Order book imbalance (top 5/10/25 levels)
- Depth imbalance
- Trade aggressor side (buy vs sell volume)

---

## 3. Interpretation Strategies (All Required)

- Multi-timeframe confluence (1m → 1d) of momentum / mean-reversion / breakout
- Order-book imbalance + absorption / exhaustion detection
- On-chain divergence vs price (flows, SOPR, MVRV Z)
- Funding rate + OI extremes as reversal / continuation signals
- Liquidation cascade detection as high-probability reversal
- Market regime classification (HMM or GMM: trend / range / high-vol / low-vol)
- Cross-asset lead-lag and rolling beta
- Sentiment spike + volume confirmation filters
- Volatility regime switching (GARCH + realized volatility)
- Statistical arbitrage / pairs trading signals (BTC-ETH etc.)
- Feature-engineered ML labels (future return quantiles 1h/4h/1d)

All strategies must be pure functions that accept a feature matrix and return a signal series (+1 / 0 / -1) + confidence.

---

## 4. Self-Improvement Stack (Mandatory)

- Walk-forward analysis
- Combinatorial purged cross-validation (to prevent leakage)
- Bayesian optimization (Optuna) + massive parallel trials (Ray)
- Genetic algorithm option for discrete strategy rules
- Reinforcement learning option (stable-baselines3) for dynamic position sizing / strategy selection
- Online / continual learning on new paper + live results
- Ensemble of strategies with dynamic weight allocation by recent rolling Sharpe / Sortino / expectancy
- Monte-Carlo path simulation for robustness testing
- Automatic feature importance pruning + adversarial robustness checks
- Nightly Celery job that:
  1. Pulls latest data
  2. Regenerates features
  3. Runs Optuna studies
  4. Backtests top candidates
  5. Promotes new ParamSet only if it beats current live one on out-of-sample metrics

---

## 5. Exact Tech Stack (Do Not Deviate)

```text
Core
- Django 5.x + Django REST Framework
- PostgreSQL + TimescaleDB extension
- Redis 7+
- Celery + django-celery-beat + django-celery-results
- Django Channels + channels-redis

Data & Compute
- ccxt.pro (real-time + REST)
- polars (primary dataframe engine – faster than pandas)
- pandas-ta
- vectorbt (primary backtester)
- web3.py
- snscrape (or twscrape) + sentence-transformers
- LightGBM + scikit-learn
- Optuna + Ray[default]
- stable-baselines3 (optional RL path)

Monitoring
- structlog or loguru
- prometheus-client
- Django admin + custom HTMX dashboard
```

Install notes for coding agent:
```bash
pip install ccxt[pro] polars pandas-ta vectorbt web3 snscrape sentence-transformers lightgbm scikit-learn optuna "ray[default]" stable-baselines3 channels channels-redis celery redis django-celery-beat django-celery-results timescaledb
```

---

## 6. Recommended App Structure

```text
trading_bot/
├── __init__.py
├── apps.py
├── models.py                 # All DB models
├── admin.py
├── urls.py
├── views.py                  # Dashboard + API
├── consumers.py              # Channels live feeds
├── tasks.py                  # All Celery tasks
├── signals.py
├── services/
│   ├── __init__.py
│   ├── data/
│   │   ├── historical.py
│   │   ├── realtime.py
│   │   ├── onchain.py
│   │   ├── sentiment.py
│   │   └── cross_asset.py
│   ├── features/
│   │   ├── engine.py         # Master feature builder (polars)
│   │   ├── technical.py
│   │   ├── orderbook.py
│   │   ├── onchain_features.py
│   │   └── sentiment_features.py
│   ├── strategies/
│   │   ├── base.py
│   │   ├── technical.py
│   │   ├── onchain.py
│   │   ├── microstructure.py
│   │   ├── regime.py
│   │   ├── ml_strategy.py
│   │   └── ensemble.py
│   ├── backtester/
│   │   ├── vectorbt_engine.py
│   │   ├── walk_forward.py
│   │   └── metrics.py
│   ├── optimizer/
│   │   ├── optuna_study.py
│   │   ├── ray_parallel.py
│   │   └── promoter.py
│   └── executor/
│       ├── paper.py
│       ├── live.py
│       ├── risk.py
│       └── position_sizer.py
├── management/
│   └── commands/
│       ├── download_history.py
│       ├── rebuild_features.py
│       └── run_full_optimization.py
└── tests/
```

---

## 7. Critical Models (models.py)

Implement at minimum:

```python
class Strategy(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=False)
    strategy_class = models.CharField(max_length=200)  # import path

class ParamSet(models.Model):
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    params = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_live = models.BooleanField(default=False)
    is_candidate = models.BooleanField(default=False)
    metrics = models.JSONField(default=dict)  # sharpe, win_rate, etc.

class BacktestRun(models.Model):
    param_set = models.ForeignKey(ParamSet, on_delete=models.CASCADE)
    start_date = models.DateTimeField()
    end_date = models.DateTimeField()
    metrics = models.JSONField()
    equity_curve = models.JSONField()  # or store as file
    created_at = models.DateTimeField(auto_now_add=True)

class FeatureSnapshot(models.Model):
    # Timescale hypertable
    timestamp = models.DateTimeField(db_index=True)
    symbol = models.CharField(max_length=20)
    features = models.JSONField()  # or better: separate numeric columns + compression

class Signal(models.Model):
    timestamp = models.DateTimeField()
    symbol = models.CharField(max_length=20)
    strategy = models.ForeignKey(Strategy, on_delete=models.CASCADE)
    param_set = models.ForeignKey(ParamSet, on_delete=models.CASCADE)
    direction = models.SmallIntegerField()  # 1, 0, -1
    confidence = models.FloatField()
    features_used = models.JSONField()

class PaperTrade / LiveTrade(models.Model):
    # full trade lifecycle
```

Use TimescaleDB hypertables for OHLCV, orderbook snapshots, and FeatureSnapshot.

---

## 8. Phased Implementation Plan

### Phase 0 – Scaffold (Day 1)
- Create app `trading_bot`
- Add to INSTALLED_APPS
- Create all models + migrations
- Basic admin registration
- urls.py + empty views
- Acceptance: `python manage.py check` passes, models appear in admin

### Phase 1 – Historical Data Pipeline
- `services/data/historical.py` using ccxt
- Management command `download_history`
- Store OHLCV in Timescale hypertable (symbol, timeframe, timestamp, o,h,l,c,v)
- Incremental update support
- Acceptance: Can download 2+ years of 1m BTCUSDT data cleanly

### Phase 2 – Real-time Ingestion
- Channels consumer for Binance websocket (orderbook, trades, funding, mark price)
- Redis streams as buffer
- On-chain poller task (every 30–60s)
- Sentiment scraper task (every 5 min)
- Acceptance: Live orderbook depth updating in Redis, visible via management command

### Phase 3 – Feature Engine
- Master `features/engine.py` that takes raw data → polars LazyFrame of all features
- Every data source from Section 2 must produce at least one feature
- Continuous aggregate or materialized view for speed
- Version features (feature_set_version)
- Acceptance: Can generate full feature matrix for any historical window in < 30s for 1 year of 1m data

### Phase 4 – Strategy Layer
- Base abstract Strategy class
- Implement every strategy from Section 3 as separate modules
- Ensemble strategy that takes weighted combination
- All strategies pure + unit tested
- Acceptance: Given a feature matrix, every strategy returns valid signal series

### Phase 5 – Backtesting Engine
- `vectorbt_engine.py` as primary
- Custom polars fallback for complex logic
- Walk-forward + combinatorial purged CV implementation
- Full metrics suite (win rate, profit factor, Sharpe, Sortino, Calmar, max DD, expectancy, etc.)
- Store every BacktestRun
- Acceptance: Can run a full walk-forward on 2 years of data and store results

### Phase 6 – Optimizer & Self-Improvement
- Optuna study definition per strategy
- Ray for parallel trials
- Promoter logic: only promote if out-of-sample metrics beat current live ParamSet by threshold
- Nightly Celery beat task that runs the full loop
- Acceptance: One full optimization cycle completes and can promote a new ParamSet

### Phase 7 – Paper Trading
- Realistic paper executor using live orderbook for fill simulation + fees + slippage
- Position tracking, PnL, risk limits
- Acceptance: Paper account can run for 24h without errors and records all signals/trades

### Phase 8 – Live Executor + Risk
- Risk engine (max position size, max daily loss, correlation limits, kill switch)
- Live order placement via existing Binance connection
- Manual promote from paper → live
- Emergency flatten endpoint
- Acceptance: Can place a tiny live order and track it end-to-end

### Phase 9 – Dashboard & Observability
- HTMX + Alpine.js live dashboard (equity, open signals, leaderboard of ParamSets, recent backtests)
- Prometheus metrics
- Full structured logging of every decision
- Acceptance: Dashboard shows live state and historical performance

### Phase 10 – Hardening & Production
- Circuit breakers on all external APIs
- Idempotent tasks
- Comprehensive test suite (unit + integration)
- Config via environment + YAML
- Documentation inside code (docstrings)
- Acceptance: System can run unattended for 7 days

---

## 9. Coding Agent Rules

1. Never hard-code API keys or secrets.
2. Every external call must have timeout + retry + circuit breaker.
3. Prefer polars over pandas everywhere possible.
4. All timestamps timezone-aware (UTC).
5. Log every signal generation and every parameter promotion.
6. Write tests for every strategy and the feature engine.
7. Use type hints + pydantic for config validation where useful.
8. Keep services pure – side effects only in tasks and executors.
9. When in doubt, make it configurable via ParamSet or YAML.

---

**End of Plan**  
Coding agent: Start with Phase 0 and proceed sequentially. After each phase, confirm acceptance criteria before moving to the next.