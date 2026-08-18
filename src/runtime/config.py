"""Runtime configuration for the public portfolio edition.

This module intentionally contains infrastructure and environment wiring only.
Production strategy thresholds, asset-specific admission rules, sizing logic,
historical setup families, and proprietary execution policies are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional for library consumers
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


DEFAULT_GAMMA_API_URL = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_API_URL = "https://clob.polymarket.com"
DEFAULT_MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
DEFAULT_CHAIN_ID = 137


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_csv(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return default
    items = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    return items or default


@dataclass(frozen=True)
class Credentials:
    """Secrets required by live execution.

    Values are loaded from environment variables and are never stored in source.
    """

    private_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_PRIVATE_KEY", ""), repr=False)
    api_key: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_KEY", ""), repr=False)
    api_secret: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_SECRET", ""), repr=False)
    api_passphrase: str = field(default_factory=lambda: os.getenv("POLYMARKET_API_PASSPHRASE", ""), repr=False)
    funder_address: str = field(default_factory=lambda: os.getenv("POLYMARKET_FUNDER", ""))
    signature_type: int = field(default_factory=lambda: _env_int("POLYMARKET_SIGNATURE_TYPE", 2))

    @property
    def live_ready(self) -> bool:
        return all((self.private_key, self.api_key, self.api_secret, self.api_passphrase))


@dataclass(frozen=True)
class NotificationSettings:
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""), repr=False)
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""), repr=False)

    @property
    def enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)


@dataclass(frozen=True)
class NetworkSettings:
    gamma_api_url: str = field(default_factory=lambda: os.getenv("PM_GAMMA_API_URL", DEFAULT_GAMMA_API_URL))
    clob_api_url: str = field(default_factory=lambda: os.getenv("PM_CLOB_API_URL", DEFAULT_CLOB_API_URL))
    market_ws_url: str = field(default_factory=lambda: os.getenv("PM_MARKET_WS_URL", DEFAULT_MARKET_WS_URL))
    chain_id: int = field(default_factory=lambda: _env_int("PM_CHAIN_ID", DEFAULT_CHAIN_ID))

    connect_timeout_seconds: float = field(default_factory=lambda: _env_float("PM_CONNECT_TIMEOUT_SEC", 3.0))
    read_timeout_seconds: float = field(default_factory=lambda: _env_float("PM_READ_TIMEOUT_SEC", 5.0))
    retry_attempts: int = field(default_factory=lambda: _env_int("PM_RETRY_ATTEMPTS", 2))
    retry_backoff_seconds: float = field(default_factory=lambda: _env_float("PM_RETRY_BACKOFF_SEC", 0.25))


@dataclass(frozen=True)
class MarketDataSettings:
    """Public market-data scope.

    These values describe which feeds the portfolio edition may observe; they are
    not production admission criteria or strategy thresholds.
    """

    assets: Tuple[str, ...] = field(
        default_factory=lambda: _env_csv(
            "PM_MARKETS",
            ("bitcoin", "ethereum", "solana", "xrp"),
        )
    )
    intervals_minutes: Tuple[int, ...] = (15,)
    discovery_cache_seconds: float = field(default_factory=lambda: _env_float("PM_DISCOVERY_CACHE_SEC", 60.0))


@dataclass(frozen=True)
class RuntimeSettings:
    """Process-level behavior for the portfolio edition."""

    paper_mode: bool = field(default_factory=lambda: _env_bool("PM_PAPER_MODE", True))
    live_trading_enabled: bool = field(default_factory=lambda: _env_bool("PM_LIVE_TRADING_ENABLED", False))
    detailed_logs: bool = field(default_factory=lambda: _env_bool("PM_DETAILED_LOGS", False))

    @property
    def live_execution_requested(self) -> bool:
        return self.live_trading_enabled and not self.paper_mode


@dataclass(frozen=True)
class AppSettings:
    credentials: Credentials = field(default_factory=Credentials)
    notifications: NotificationSettings = field(default_factory=NotificationSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    market_data: MarketDataSettings = field(default_factory=MarketDataSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    def validate(self) -> None:
        """Fail clearly if live execution is requested without credentials."""
        if self.runtime.live_execution_requested and not self.credentials.live_ready:
            raise RuntimeError(
                "Live execution was requested, but Polymarket credentials are incomplete."
            )


settings = AppSettings()
