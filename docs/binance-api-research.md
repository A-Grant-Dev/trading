# Binance API Research — Trading Analysis & Prediction System

> **Date:** July 14, 2026  
> **Source:** [developers.binance.com/en/docs](https://developers.binance.com/en/docs)  
> **Historical Data:** [data.binance.vision](https://data.binance.vision/)

---

## Important Note

Binance does **not** provide any built-in prediction, signal generation, or technical indicator APIs. There is no endpoint that returns RSI, MACD, or price predictions. You must pull raw data and compute everything client-side using libraries like `TA-Lib` or `pandas-ta`.

---

## 1. Core Price Data (The Foundation)

### Spot Market Data Endpoints

| Endpoint | Path | Description | Prediction Use |
|:---------|:-----|:------------|:---------------|
| **Klines/Candles** | `GET /api/v3/klines` | OHLCV + volume + taker buy/sell by interval | Chart patterns, indicators, ML training features |
| **Aggregate Trades** | `GET /api/v3/aggTrades` | Aggregated trade history | Volume profile, market micro-structure |
| **Recent Trades** | `GET /api/v3/trades` | Raw list of recent trades | Tick-level analysis |
| **Order Book Depth** | `GET /api/v3/depth` | Top bid/ask levels with quantities | Order book imbalance, support/resistance, liquidity |
| **24hr Ticker** | `GET /api/v3/ticker/24hr` | Price change %, volume, high/low, last price | Trend strength, volatility snapshot |
| **Symbol Price** | `GET /api/v3/ticker/price` | Latest price for all or single symbol | Current market value |
| **Exchange Info** | `GET /api/v3/exchangeInfo` | Trading rules, symbol filters, precision | Required for any trading automation |

### Available Kline Intervals

```
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
```

### REST Limit

- Max **1000 candles** per request via REST API
- For ML training data: use **[data.binance.vision](https://data.binance.vision/)** — bulk historical CSV downloads going back years

---

## 2. Futures-Specific Data (Highest Value — Market Sentiment)

These are the **most valuable signals** for prediction. Most retail traders don't have access to this data.

| Endpoint | Path | What It Tells You |
|:---------|:-----|:------------------|
| **Funding Rate** | `GET /fapi/v1/fundingInfo` | Who's paying whom? Predicts long squeezes vs short squeezes. Positive funding = longs paying shorts (bearish signal when extreme). |
| **Open Interest** | `GET /fapi/v1/openInterest` | Total outstanding contracts. Rising OI = strong trend. Divergence between price & OI = potential reversal. |
| **Top Trader L/S Ratio** | `GET /futures/data/topLongShortAccountRatio` | Whale positioning — contrarian signal when extreme. |
| **Global L/S Ratio** | `GET /futures/data/globalLongShortAccountRatio` | Overall market sentiment. When everyone is long, expect a dump. |
| **Taker Buy/Sell Volume** | Included in ticker responses | Who's aggressive — buyers or sellers? Shows real market direction. |
| **Liquidation History** | `GET /futures/data/liquidationHist` | Cascade events. Massive liquidations in one direction often precede violent reversals. |

### Why Futures Data Matters for Prediction

```
Funding Rate + L/S Ratio + Open Interest + Liquidation Data
    ↓
Market Sentiment Score (bullish / bearish / extreme)
    ↓
Contrarian Signal → "When everyone is long, the top is near"
```

---

## 3. Real-Time Data (WebSocket Streams)

For low-latency prediction models, use WebSocket streams instead of REST polling.

| Stream | Path Pattern | Use Case |
|:-------|:-------------|:---------|
| **Kline Stream** | `<symbol>@kline_<interval>` | Real-time candle updates — compute indicators on every new close |
| **Depth Stream** | `<symbol>@depth` or `<symbol>@depth20@100ms` | Real-time order book for imbalance detection |
| **Agg Trade Stream** | `<symbol>@aggTrade` | Every trade as it happens — volume tracking |
| **Ticker Stream** | `<symbol>@ticker` | Rolling 24hr stats updated every second |
| **All Market Tickers** | `!ticker@arr` | All tickers in one stream |
| **Mini Ticker** | `<symbol>@miniTicker` | Lighter ticker (only close + volume) |

### WebSocket URLs

- **Spot:** `wss://stream.binance.com:9443/ws/<stream_name>`
- **Futures:** `wss://fstream.binance.com/ws/<stream_name>`
- **Combined:** `wss://stream.binance.com:9443/stream?streams=<stream1>/<stream2>`

---

## 4. Computation Layer (Client-Side)

Since Binance doesn't provide indicators, you need to compute them yourself using:

| Library | Language | Notes |
|:--------|:---------|:------|
| **TA-Lib** | C/C++, Python bindings | Fastest option. 150+ indicators (RSI, MACD, Bollinger, Stochastic, ATR, Ichimoku, etc.) |
| **pandas-ta** | Pure Python | Easier to use, 130+ indicators, integrates with pandas DataFrames |
| **ta** | Pure Python | Another lightweight technical analysis library |

### Key Indicators for Prediction

- **Trend:** SMA, EMA, MACD, Ichimoku Cloud, ADX
- **Momentum:** RSI, Stochastic, Williams %R, CCI
- **Volatility:** Bollinger Bands, ATR, Keltner Channels
- **Volume:** OBV, Volume Profile, VWAP, Money Flow Index
- **Patterns:** Engulfing, Doji, Hammer, Morning/Evening Star (TA-Lib has candlestick pattern recognition)

---

## 5. System Architecture for Prediction

```
┌───────────────────────────────────────────────────────────────┐
│                      DATA LAYER                               │
│                                                               │
│  REST: Klines (OHLCV) → Historical training data              │
│  WS:   Kline Stream    → Live candle updates                  │
│  WS:   Depth Stream    → Order book imbalance                 │
│  WS:   Trade Stream    → Volume / aggression tracking         │
│  REST: Funding Rate    → Market skew (Futures)                │
│  REST: L/S Ratios      → Sentiment (Futures)                  │
│  REST: Open Interest   → Trend strength (Futures)             │
│  REST: Liquidations    → Cascade risk (Futures)               │
└───────────────────────────┬───────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING                          │
│                                                               │
│  pandas-ta / TA-Lib:                                          │
│  → RSI, MACD, Bollinger Bands, ATR, Stochastic                │
│  → Volume Profile, VWAP, OBV                                  │
│  → Candlestick Pattern Recognition                            │
│  → Order Book Imbalance Ratio                                 │
│  → Funding Rate Sentiment Score                               │
│  → L/S Ratio Extremes Detection                               │
└───────────────────────────┬───────────────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────────────┐
│                   PREDICTION MODELS                            │
│                                                               │
│  Option A: ML Models                                          │
│  → LSTM / GRU (time series forecasting)                       │
│  → XGBoost / LightGBM (feature-based classification)          │
│  → Transformer Models (advanced sequence modeling)            │
│                                                               │
│  Option B: Rule-Based Signals                                 │
│  → Divergence detection (RSI/price divergence)                │
│  → Support/Resistance breakouts                               │
│  → Liquidation cascade triggers                               │
│  → Funding rate extreme reversals                             │
└───────────────────────────────────────────────────────────────┘
```

---

## 6. Rate Limits & Best Practices

- Each endpoint consumes "weight" — exceeding limits results in IP ban
- **WebSockets are preferred** for real-time to avoid REST rate limits
- Use `exchangeInfo` endpoint to get symbol filters (LOT_SIZE, PRICE_FILTER, MIN_NOTIONAL) — critical if automating trades
- For historical ML training data, download CSVs from `data.binance.vision` instead of using REST

---

## 7. Quick Reference — API Base URLs

| Service | Base URL |
|:--------|:---------|
| Spot REST | `https://api.binance.com` |
| Futures REST | `https://fapi.binance.com` |
| Spot WebSocket | `wss://stream.binance.com:9443` |
| Futures WebSocket | `wss://fstream.binance.com` |
| Historical Data | `https://data.binance.vision` |
| Developer Docs | `https://developers.binance.com/en/docs` |

---

---

## 8. Binance Algo Trading Product (TWAP / VWAP / Iceberg Orders)

> **Source:** [developers.binance.com/en/docs/products/algo/quick-start](https://developers.binance.com/en/docs/products/algo/quick-start)

### What Is It?

Binance's **Algo Trading** product is a set of **automated execution algorithms** designed for large-volume traders (institutions, whales) to buy or sell massive amounts of crypto without moving the price against themselves. Instead of dumping a 500 BTC order into the order book all at once (which would crash the price), the algo breaks it into hundreds of tiny orders over time.

There are **no extra fees** — you pay the same maker/taker fees as normal trades. No additional subscription cost.

---

### The 3 Algorithm Types

#### 1. TWAP (Time-Weighted Average Price)

| Aspect | Detail |
|:-------|:-------|
| **Purpose** | Execute a large order evenly over a set time period |
| **How it works** | Splits the total order into equal time slices — trades X amount every N seconds/minutes regardless of volume |
| **Best for** | **Accumulation/Distribution** — slowly building or exiting a large position without spooking the market |
| **When NOT to use** | Low liquidity pairs (price might drift too far) |
| **Example** | Buy 100 BTC over 4 hours = ~0.4 BTC every minute automatically |
| **API Endpoint** | `POST /sapi/v1/algo/spot/newOrderTwap` |

**Key parameters:** `symbol`, `side` (BUY/SELL), `quantity`, `duration` (total time), optionally a `limitPrice` to cap max price.

#### 2. VWAP (Volume-Weighted Average Price)

| Aspect | Detail |
|:-------|:-------|
| **Purpose** | Execute the order to match the volume-weighted average market price |
| **How it works** | Unlike TWAP (time-based), VWAP is **volume-aware** — it trades more during high-volume periods and less during low-volume periods, naturally blending into market activity |
| **Best for** | Getting a "fair" execution price that mirrors overall market participation |
| **When NOT to use** | Thin markets where volume is very uneven |
| **Example** | Buy $500K worth of ETH — algorithm watches market volume and adjusts trade size dynamically |
| **API Endpoint** | `POST /sapi/v1/algo/spot/newOrderVwap` |

#### 3. Iceberg Orders

| Aspect | Detail |
|:-------|:-------|
| **Purpose** | Hide the true size of your order from the order book |
| **How it works** | Only a small "tip" of your order shows on the book. As each chunk fills, another appears — your full size stays hidden |
| **Best for** | **Distribution** — selling large amounts without revealing your hand |
| **When NOT to use** | When you WANT to show size (e.g., to intimidate short sellers) |
| **Example** | Place a sell order for 10,000 ETH but only 50 ETH shows on the book at a time |
| **API Endpoint** | `POST /sapi/v1/algo/spot/newOrderIceberg` |

---

### All Algo API Endpoints

#### Spot Algo (`/sapi/v1/algo/spot/`)

| Method | Endpoint | Purpose |
|:-------|:---------|:--------|
| `POST` | `/sapi/v1/algo/spot/newOrderTwap` | Create a new TWAP order |
| `POST` | `/sapi/v1/algo/spot/newOrderVwap` | Create a new VWAP order |
| `POST` | `/sapi/v1/algo/spot/newOrderIceberg` | Create a new Iceberg order |
| `DELETE` | `/sapi/v1/algo/spot/order` | Cancel an active algo order |
| `GET` | `/sapi/v1/algo/spot/openOrders` | List all active algo orders |
| `GET` | `/sapi/v1/algo/spot/order` | Query a specific algo order status |
| `GET` | `/sapi/v1/algo/spot/historicalOrders` | View past algo orders |

#### Futures Algo (`/sapi/v1/algo/futures/`)

The same endpoints exist for USDS-M Futures under `/algo/futures/`.

---

### How It Connects to Prediction

This is the **execution layer** of the whole system:

```
Prediction Signal ("BTC will go up 5% in the next hour")
    ↓
Decision: Buy $100K worth of BTC
    ↓
Execution Strategy: Use TWAP over 4 hours to avoid slippage
    ↓
API Call: POST /sapi/v1/algo/spot/newOrderTwap
    ↓
Binance executes ~$400 of BTC every minute automatically
    ↓
Result: You're fully positioned with minimal market impact
```

Without algo trading, a $100K market buy would instantly spike the price 1-2% — you'd lose thousands to slippage. With TWAP/VWAP, that slippage is virtually eliminated.

---

### Real-World Strategy Examples

| Strategy | What You Do | Algo Type | Why |
|:---------|:------------|:----------|:----|
| **Accumulate during dip** | You predict BTC will recover so you want to buy $50K over 8 hours | **TWAP** | Spreads buys evenly to avoid pumping the price back up |
| **Sell into strength** | A coin is pumping 30% — you want to sell your bag | **VWAP** | Sells more when volume is highest (when others are also buying) |
| **Hide a massive sell wall** | You hold 1% of a coin's supply and need to sell | **Iceberg** | Prevents everyone from seeing your giant sell order and panicking |
| **Dollar-cost average** | Buy $1K of ETH daily for 30 days | **TWAP** | Set and forget — automated DCA at the exchange level |

---

### Requirements & Restrictions

| Requirement | Detail |
|:------------|:-------|
| **Minimum Order Size** | Varies by symbol — typically higher than normal orders |
| **Minimum Duration** | Usually a minimum time window (e.g., 5 minutes minimum for TWAP) |
| **Maximum Duration** | Usually capped (e.g., 24 hours max) |
| **Balance** | Must have sufficient balance locked up front |
| **Rate Limits** | Algo endpoints have their own rate limit weight |
| **Market Hours** | Only works during active market hours for that pair |
| **No Fill Guarantee** | If liquidity dries up, the algo may not complete |

---

*End of research document. Last updated: July 14, 2026.*
