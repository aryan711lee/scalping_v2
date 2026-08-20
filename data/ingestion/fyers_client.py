import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import pytz

import app.logger  # noqa: F401 – side-effect: configures logging
from app.config import (
    FYERS_APP_ID,
    FYERS_ACCESS_TOKEN,
    API_RETRY_COUNT,
    API_RETRY_BACKOFF,
)

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")


class FyersClient:
    def __init__(self) -> None:
        self._fyers = None
        self._init_client()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        try:
            from fyers_apiv3 import fyersModel  # type: ignore

            self._fyers = fyersModel.FyersModel(
                client_id=FYERS_APP_ID,
                token=FYERS_ACCESS_TOKEN,
                log_path="",
                is_async=False,
            )
            logger.info("Fyers client initialised (app_id=%s)", FYERS_APP_ID)
        except ImportError as exc:
            logger.error("fyers-apiv3 not installed: %s", exc)
            raise
        except Exception as exc:
            logger.error("Failed to initialise Fyers client: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_session_valid(self) -> bool:
        try:
            resp = self._fyers.get_profile()
            return resp.get("code") == 200
        except Exception as exc:
            logger.warning("Session validity check failed: %s", exc)
            return False

    def get_historical_candles(
        self,
        symbol: str,
        resolution: str,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame:
        """
        Returns DataFrame: datetime (IST-aware), open, high, low, close, volume.
        Handles date-range chunking automatically for long ranges.
        """
        from_dt = datetime.strptime(date_from, "%Y-%m-%d")
        to_dt = datetime.strptime(date_to, "%Y-%m-%d")

        chunks = self._build_date_chunks(from_dt, to_dt, int(resolution))
        frames: list[pd.DataFrame] = []

        for chunk_from, chunk_to in chunks:
            df = self._fetch_chunk_with_retry(symbol, resolution, chunk_from, chunk_to)
            if df is not None and not df.empty:
                frames.append(df)
            time.sleep(0.5)  # rate-limit guard between chunks

        if not frames:
            logger.warning("No data returned for %s [%s] %s – %s", symbol, resolution, date_from, date_to)
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        result = pd.concat(frames, ignore_index=True)
        result.drop_duplicates(subset=["datetime"], keep="first", inplace=True)
        result.sort_values("datetime", inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result

    def get_live_feed_stub(self, symbols: list[str]) -> None:
        """Stub — live websocket feed to be implemented in Phase 3."""
        logger.info("Live feed stub called for %d symbols (not yet implemented)", len(symbols))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_date_chunks(
        self,
        from_dt: datetime,
        to_dt: datetime,
        resolution_minutes: int,
    ) -> list[tuple[str, str]]:
        """
        Fyers caps intraday history at 100 days per request for resolutions
        1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 120, 180, 240 minutes.
        Use 90-day chunks to stay safely under the limit.
        """
        chunk_days = 90

        chunks = []
        cursor = from_dt
        while cursor <= to_dt:
            end = min(cursor + timedelta(days=chunk_days - 1), to_dt)
            chunks.append((cursor.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
            cursor = end + timedelta(days=1)
        return chunks

    def _fetch_chunk_with_retry(
        self,
        symbol: str,
        resolution: str,
        date_from: str,
        date_to: str,
    ) -> pd.DataFrame | None:
        payload = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",  # YYYY-MM-DD strings
            "range_from": date_from,
            "range_to": date_to,
            "cont_flag": "1",
        }

        backoff = API_RETRY_BACKOFF
        for attempt in range(1, API_RETRY_COUNT + 1):
            try:
                resp = self._fyers.history(data=payload)
                if resp.get("code") == 200 and resp.get("candles"):
                    return self._parse_candles(resp["candles"])
                else:
                    logger.warning(
                        "Attempt %d/%d: unexpected response for %s [%s – %s]: %s",
                        attempt, API_RETRY_COUNT, symbol, date_from, date_to, resp,
                    )
            except Exception as exc:
                logger.warning(
                    "Attempt %d/%d: exception fetching %s [%s – %s]: %s",
                    attempt, API_RETRY_COUNT, symbol, date_from, date_to, exc,
                )

            if attempt < API_RETRY_COUNT:
                time.sleep(backoff)
                backoff *= 2

        logger.error(
            "All %d attempts failed for %s [%s – %s]",
            API_RETRY_COUNT, symbol, date_from, date_to,
        )
        return None

    @staticmethod
    def _parse_candles(candles: list) -> pd.DataFrame:
        df = pd.DataFrame(candles, columns=["datetime", "open", "high", "low", "close", "volume"])

        # Fyers returns epoch seconds in UTC
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True).dt.tz_convert(IST)

        df["open"] = df["open"].astype("float64")
        df["high"] = df["high"].astype("float64")
        df["low"] = df["low"].astype("float64")
        df["close"] = df["close"].astype("float64")
        df["volume"] = df["volume"].astype("int64")

        return df
