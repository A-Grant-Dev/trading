"""
Historical Data Ingestion Service

Downloads historical kline data from Binance REST API and the official
Binance public data repository (data.binance.vision), then imports into
the MarketData model for backtesting and ML training.

Sources:
  - Binance REST API (klines endpoint): Recent historical data
  - data.binance.vision: Full historical CSV archives for deep backtesting

Renaissance principle: High-quality, high-resolution historical data
is the foundation of every quantitative model. Garbage in = garbage out.
"""

import calendar
import csv
import io
import logging
import time
import zipfile
from datetime import date, datetime, timezone
from decimal import Decimal

import requests

from quant.models import HistoricalDataBatch, MarketData

logger = logging.getLogger(__name__)

BINANCE_API = "https://api.binance.com"
BINANCE_DATA_VISION = "https://data.binance.vision"
REQUEST_TIMEOUT = 30


# ── Binance REST Klines ────────────────────────────────────────────


def fetch_klines_rest(
    symbol: str,
    interval: str,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = 1000,
) -> list[dict]:
    """
    Fetch kline/candlestick data from Binance REST API.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Kline interval (1m, 5m, 1h, etc.)
        start_time: Milliseconds since epoch (optional)
        end_time: Milliseconds since epoch (optional)
        limit: Max 1000 per request

    Returns:
        List of parsed kline dicts
    """
    url = f"{BINANCE_API}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    try:
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch klines for {symbol} {interval}: {e}")
        raise

    klines = []
    for k in raw:
        klines.append({
            "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
            "open": Decimal(str(k[1])),
            "high": Decimal(str(k[2])),
            "low": Decimal(str(k[3])),
            "close": Decimal(str(k[4])),
            "volume": Decimal(str(k[5])),
            "close_time": datetime.fromtimestamp(k[6] / 1000, tz=timezone.utc),
            "quote_asset_volume": Decimal(str(k[7])),
            "trades": int(k[8]),
            "taker_buy_base_vol": Decimal(str(k[9])),
            "taker_buy_quote_vol": Decimal(str(k[10])),
            "closed": True,
        })
    return klines


def import_klines_to_marketdata(
    symbol: str,
    interval: str,
    klines: list[dict],
    batch: HistoricalDataBatch | None = None,
) -> int:
    """
    Bulk-insert klines into MarketData table.

    Uses bulk_create with ignore_conflicts for performance.
    Returns the number of new rows inserted.

    Args:
        symbol: Trading pair
        interval: Kline interval
        klines: List of kline dicts from fetch_klines_rest()
        batch: Optional HistoricalDataBatch to update with row count

    Returns:
        Number of new records created
    """
    if not klines:
        return 0

    objects = []
    for k in klines:
        objects.append(MarketData(
            symbol=symbol.upper(),
            interval=interval,
            open_time=k["open_time"],
            open=k["open"],
            high=k["high"],
            low=k["low"],
            close=k["close"],
            volume=k["volume"],
            quote_asset_volume=k.get("quote_asset_volume"),
            taker_buy_base_vol=k.get("taker_buy_base_vol"),
            taker_buy_quote_vol=k.get("taker_buy_quote_vol"),
            trades=k.get("trades"),
            closed=k.get("closed", True),
        ))

    created = MarketData.objects.bulk_create(
        objects,
        ignore_conflicts=True,  # Skip duplicates
        batch_size=500,
    )

    count = len(created)
    if batch:
        batch.rows_imported = (batch.rows_imported or 0) + count
        batch.save(update_fields=["rows_imported"])

    logger.info(f"Imported {count} new {symbol} {interval} klines")
    return count


def download_historical_range(
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str | None = None,
) -> int:
    """
    Download and import a range of historical klines using the REST API.

    Walks through time in 1000-candle chunks to cover the full range.

    Args:
        symbol: Trading pair
        interval: Kline interval
        start_str: Start date (e.g., '2024-01-01' or ISO format)
        end_str: End date (default: now)

    Returns:
        Total rows imported
    """
    start_dt = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)
        if end_str
        else datetime.now(timezone.utc)
    )

    batch = HistoricalDataBatch.objects.create(
        symbol=symbol.upper(),
        interval=interval,
        date_range_start=start_dt.date(),
        date_range_end=end_dt.date(),
        status="downloading",
    )

    # Interval → millisecond duration mapping for chunking
    interval_ms = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
        "3d": 259_200_000,
        "1w": 604_800_000,
        "1M": 2_592_000_000,
    }.get(interval, 60_000)

    total = 0
    current_start = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    chunk_size = interval_ms * 1000  # 1000 candles per request

    batch.status = "importing"
    batch.save(update_fields=["status"])

    try:
        while current_start < end_ms:
            klines = fetch_klines_rest(
                symbol, interval,
                start_time=current_start,
                limit=1000,
            )
            if not klines:
                break

            total += import_klines_to_marketdata(symbol, interval, klines, batch)

            # Advance to next chunk (use last candle's open_time + 1ms)
            last_time = int(klines[-1]["open_time"].timestamp() * 1000)
            current_start = last_time + 1

            # Avoid rate limiting
            time.sleep(0.1)

        batch.status = "completed"
        batch.completed_at = datetime.now(timezone.utc)
        batch.save(update_fields=["status", "completed_at"])

    except Exception as e:
        batch.status = "failed"
        batch.error_message = str(e)
        batch.save(update_fields=["status", "error_message"])
        logger.exception(f"Historical download failed for {symbol} {interval}")
        raise

    logger.info(f"Historical import complete: {symbol} {interval}: {total} rows")
    return total


# ── data.binance.vision CSV Import ─────────────────────────────────


def _parse_binance_csv_line(line: str) -> dict | None:
    """
    Parse a single line from Binance's monthly kline CSV format.

    CSV format (no header):
        open_time (Unix ms), open, high, low, close, volume,
        close_time, quote_vol, trades, taker_buy_base_vol,
        taker_buy_quote_vol, ignore
    """
    try:
        parts = list(csv.reader([line]))[0]
        if len(parts) < 11:
            return None

        return {
            "open_time": datetime.fromtimestamp(int(parts[0]) / 1000, tz=timezone.utc),
            "open": Decimal(parts[1]),
            "high": Decimal(parts[2]),
            "low": Decimal(parts[3]),
            "close": Decimal(parts[4]),
            "volume": Decimal(parts[5]),
            "quote_asset_volume": Decimal(parts[7]) if parts[7] else None,
            "trades": int(parts[8]) if parts[8] else None,
            "taker_buy_base_vol": Decimal(parts[9]) if parts[9] else None,
            "taker_buy_quote_vol": Decimal(parts[10]) if len(parts) > 10 and parts[10] else None,
            "closed": True,
        }
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse CSV line: {e}")
        return None


def download_monthly_csv(
    symbol: str,
    interval: str,
    year: int,
    month: int,
) -> int:
    """
    Download a monthly CSV zip from data.binance.vision and import.

    URL format:
        https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/
        {symbol}-{interval}-{year}-{month:02d}.zip

    Args:
        symbol: Trading pair
        interval: Kline interval
        year: Year (e.g., 2024)
        month: Month (1-12)

    Returns:
        Number of rows imported
    """
    filename = f"{symbol.upper()}-{interval}-{year}-{month:02d}"
    url = (
        f"{BINANCE_DATA_VISION}/data/spot/monthly/klines/"
        f"{symbol.upper()}/{interval}/{filename}.zip"
    )

    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT * 2)
        resp.raise_for_status()
    except requests.HTTPError as e:
        if resp.status_code == 404:
            logger.info(f"No data at {url} (maybe not available yet)")
            return 0
        logger.error(f"Failed to download {url}: {e}")
        raise

    # Parse the zip file
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_filename = f"{filename}.csv"
            if csv_filename not in zf.namelist():
                logger.warning(f"{csv_filename} not found in zip")
                return 0

            with zf.open(csv_filename) as f:
                content = f.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to parse zip for {filename}: {e}")
        raise

    # Parse CSV lines
    klines = []
    for line in content.strip().split("\n"):
        parsed = _parse_binance_csv_line(line)
        if parsed:
            klines.append(parsed)

    if not klines:
        return 0

    # Create batch record
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1)
    else:
        end_date = date(year, month, calendar.monthrange(year, month)[1])

    batch = HistoricalDataBatch.objects.create(
        symbol=symbol.upper(),
        interval=interval,
        date_range_start=start_date,
        date_range_end=end_date,
        file_path=url,
        file_size_bytes=len(resp.content),
        status="importing",
    )

    count = import_klines_to_marketdata(symbol.upper(), interval, klines, batch)

    batch.status = "completed"
    batch.completed_at = datetime.now(timezone.utc)
    batch.save(update_fields=["status", "completed_at"])

    return count


# ── Query Helper ───────────────────────────────────────────────────


def get_market_data_df(symbol: str, interval: str, limit: int = 1000) -> "pd.DataFrame":
    """
    Query MarketData table and return results as a pandas DataFrame.

    This is the primary data accessor for all quant models.
    The DataFrame has:
      - Index: open_time (datetime)
      - Columns: open, high, low, close, volume, trades, ...

    Args:
        symbol: Trading pair
        interval: Kline interval
        limit: Max rows (default 1000)

    Returns:
        DataFrame with OHLCV data, sorted chronologically
    """
    import pandas as pd

    qs = MarketData.objects.filter(
        symbol=symbol.upper(),
        interval=interval,
    ).order_by("open_time")[:limit]

    data = list(qs.values(
        "open_time", "open", "high", "low", "close",
        "volume", "quote_asset_volume", "trades",
        "taker_buy_base_vol", "taker_buy_quote_vol",
    ))

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.set_index("open_time", inplace=True)

    # Convert Decimal columns to float for pandas/numpy compatibility
    decimal_cols = ["open", "high", "low", "close", "volume",
                    "quote_asset_volume", "taker_buy_base_vol", "taker_buy_quote_vol"]
    for col in decimal_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(float)

    return df
