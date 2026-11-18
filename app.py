"""TradingView alert JSON → a public signal file on this Render service.

TradingView cancels a webhook after 3 seconds. The webhook token is never
logged and is never written into the signal file.

Render starts this with the Procfile:
uvicorn app:app --host 0.0.0.0 --port $PORT

/health and /signal.json do not call QuantConnect. There is no QC API token.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("tv_qc_bridge")
log.setLevel(logging.INFO)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    log.addHandler(_handler)
log.propagate = False

MAX_BODY_BYTES = 8192
MAX_TOKEN_LEN = 512
MAX_TEXT_LEN = 128
ALLOWED_SIDES = frozenset({"long", "short", "flat"})
# Render's disk is ephemeral. /tmp survives only until this process is replaced.
_SIGNAL_PATH = os.environ.get("SIGNAL_FILE", "/tmp/tv-signal.json")
_LOCK = threading.Lock()
_SIGNAL: dict[str, Any] = {}

# Fields written to signal.json. token and exchange are never included.
# token and exchange stay on the webhook body and are not sent to QuantConnect.
COMMAND_TEXT_KEYS = ("pattern", "signal", "ticker", "tv_symbol", "timeframe", "time")

app = FastAPI(title="tv-qc-bridge")


@dataclass(frozen=True)
class Settings:
    webhook_token: str

    def __repr__(self) -> str:
        return "Settings(redacted)"


class SettingsError(Exception):
    pass


def _clean_env(name: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        raise SettingsError(f"missing env {name}")
    value = raw.strip()
    if not value:
        raise SettingsError(f"missing env {name}")
    if "\n" in value or "\r" in value:
        raise SettingsError(f"env {name} contains a newline")
    return value


def load_settings() -> Settings:
    return Settings(webhook_token=_clean_env("WEBHOOK_TOKEN"))


def token_matches(provided: Any, expected: str) -> bool:
    """Constant-time check. Non-strings and over-long values never match."""
    if not isinstance(provided, str) or len(provided) > MAX_TOKEN_LEN:
        provided = ""
    actual = hashlib.sha256(provided.encode("utf-8")).digest()
    wanted = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(actual, wanted)


def command_from_signal(body: dict[str, Any]) -> dict[str, Any]:
    """Map qcSignalJson() into the live-command object. Drops token and exchange."""
    if body.get("source") != "tradingview":
        raise ValueError("source")
    command: dict[str, Any] = {"source": "tradingview"}
    for key in COMMAND_TEXT_KEYS:
        value = body.get(key)
        if not isinstance(value, str):
            raise ValueError(key)
        value = value.strip()
        if not value or len(value) > MAX_TEXT_LEN or "\n" in value or "\r" in value:
            raise ValueError(key)
        command[key] = value
    side = body.get("side")
    if not isinstance(side, str):
        raise ValueError("side")
    side = side.strip().lower()
    if side not in ALLOWED_SIDES:
        raise ValueError("side")
    command["side"] = side
    close = body.get("close")
    if isinstance(close, bool) or not isinstance(close, (int, float)):
        raise ValueError("close")
    command["close"] = close
    # Stable order matches the guide stub. side is inserted after signal.
    return {
        "source": command["source"],
        "pattern": command["pattern"],
        "signal": command["signal"],
        "side": command["side"],
        "ticker": command["ticker"],
        "tv_symbol": command["tv_symbol"],
        "timeframe": command["timeframe"],
        "close": command["close"],
        "time": command["time"],
    }


def _error(status: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": code})


def _read_signal_file() -> dict[str, Any]:
    try:
        with open(_SIGNAL_PATH, encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _write_signal(command: dict[str, Any]) -> None:
    """Persist the command JSON. The webhook token is not part of command."""
    global _SIGNAL
    text = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    temporary = _SIGNAL_PATH + ".tmp"
    directory = os.path.dirname(_SIGNAL_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, _SIGNAL_PATH)
    with _LOCK:
        _SIGNAL = dict(command)


def current_signal() -> dict[str, Any]:
    with _LOCK:
        if _SIGNAL:
            return dict(_SIGNAL)
    loaded = _read_signal_file()
    with _LOCK:
        if not _SIGNAL and loaded:
            _SIGNAL.update(loaded)
        return dict(_SIGNAL)


@app.get("/health")
async def health() -> dict[str, str]:
    # UptimeRobot only. Do not read secrets or call QuantConnect.
    return {"status": "ok"}


@app.get("/signal_x9k2m7q4.json")
async def signal_file() -> JSONResponse:
    # Public on purpose: the QC algorithm has no API token and can only GET.
    return JSONResponse(status_code=200, content=current_signal())


@app.post("/webhook")
async def webhook(request: Request) -> JSONResponse:
    try:
        settings = load_settings()
    except SettingsError as exc:
        log.error("%s", exc)
        return _error(500, "misconfigured")

    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        return _error(400, "bad_json")
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return _error(400, "bad_json")
    if not isinstance(body, dict):
        return _error(400, "bad_json")

    if not token_matches(body.get("token"), settings.webhook_token):
        log.info("rejected token")
        return _error(401, "unauthorized")

    try:
        command = command_from_signal(body)
    except ValueError as exc:
        field = exc.args[0] if exc.args else "payload"
        log.info("rejected payload field=%s", field)
        return _error(400, "bad_payload")

    try:
        _write_signal(command)
    except OSError:
        log.error("signal file write failed")
        return _error(500, "store")

    def _public(value: str) -> str:
        return "[redacted]" if value == settings.webhook_token else value

    log.info(
        "stored pattern=%s signal=%s side=%s ticker=%s",
        _public(command["pattern"]),
        _public(command["signal"]),
        _public(command["side"]),
        _public(command["ticker"]),
    )
    return JSONResponse(status_code=200, content={"ok": True})
