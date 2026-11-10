"""TradingView alert JSON → QuantConnect live command.

TradingView cancels a webhook after 3 seconds, so the QuantConnect POST is
capped at about 2 seconds. The webhook token and the QC API token are never
logged.

Render starts this with the Procfile:
uvicorn app:app --host 0.0.0.0 --port $PORT

/health does not load settings or call QuantConnect. The HTTP client is
imported only when a webhook is forwarded.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
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

DEFAULT_QC_URL = "https://www.quantconnect.com/api/v2/live/commands/create"
QC_TIMEOUT_SECONDS = 2.0
MAX_BODY_BYTES = 8192
MAX_TOKEN_LEN = 512
MAX_TEXT_LEN = 128
ALLOWED_SIDES = frozenset({"long", "short", "flat"})

# Fields the guide's on_command stub reads, plus the extras the bridge stub forwards.
# token and exchange stay on the webhook body and are not sent to QuantConnect.
COMMAND_TEXT_KEYS = ("pattern", "signal", "ticker", "tv_symbol", "timeframe", "time")

app = FastAPI(title="tv-qc-bridge")


@dataclass(frozen=True)
class Settings:
    webhook_token: str
    qc_user_id: str
    qc_api_token: str
    project_id: int
    qc_url: str

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
    webhook_token = _clean_env("WEBHOOK_TOKEN")
    user_id = _clean_env("QC_USER_ID")
    api_token = _clean_env("QC_API_TOKEN")
    project_raw = _clean_env("QC_PROJECT_ID")
    if not user_id.isdigit():
        raise SettingsError("QC_USER_ID must be digits")
    if not project_raw.isdigit() or int(project_raw) <= 0:
        raise SettingsError("QC_PROJECT_ID must be a positive integer")
    qc_url = os.environ.get("QC_COMMAND_URL", "").strip() or DEFAULT_QC_URL
    if "\n" in qc_url or "\r" in qc_url or not qc_url.startswith(("https://", "http://")):
        raise SettingsError("QC_COMMAND_URL must be an http(s) URL")
    return Settings(
        webhook_token=webhook_token,
        qc_user_id=user_id,
        qc_api_token=api_token,
        project_id=int(project_raw),
        qc_url=qc_url,
    )


def token_matches(provided: Any, expected: str) -> bool:
    """Constant-time check. Non-strings and over-long values never match."""
    if not isinstance(provided, str) or len(provided) > MAX_TOKEN_LEN:
        provided = ""
    actual = hashlib.sha256(provided.encode("utf-8")).digest()
    wanted = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(actual, wanted)


def qc_auth_headers(user_id: str, api_token: str) -> dict[str, str]:
    """Official hashed-token scheme: Basic base64(userId:sha256(apiToken:timestamp))."""
    timestamp = str(int(time.time()))
    digest = hashlib.sha256(f"{api_token}:{timestamp}".encode("utf-8")).hexdigest()
    encoded = base64.b64encode(f"{user_id}:{digest}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {encoded}", "Timestamp": timestamp}


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


async def post_command(url: str, headers: dict[str, str], payload: dict[str, Any]) -> Any:
    import httpx

    timeout = httpx.Timeout(QC_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
        return await client.post(url, headers=headers, json=payload)


def _qc_rejected(response: Any) -> bool:
    if response.status_code < 200 or response.status_code >= 300:
        return True
    try:
        parsed = response.json()
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and parsed.get("success") is False


@app.get("/health")
async def health() -> dict[str, str]:
    # UptimeRobot only. Do not read secrets or call QuantConnect.
    return {"status": "ok"}


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

    import httpx

    payload = {"projectId": settings.project_id, "command": command}
    headers = qc_auth_headers(settings.qc_user_id, settings.qc_api_token)
    try:
        response = await asyncio.wait_for(
            post_command(settings.qc_url, headers, payload),
            timeout=QC_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        log.warning("quantconnect timeout")
        return _error(502, "quantconnect")
    except httpx.HTTPError as exc:
        log.warning("quantconnect http error %s", type(exc).__name__)
        return _error(502, "quantconnect")

    if _qc_rejected(response):
        log.warning("quantconnect rejected status=%s", response.status_code)
        return _error(502, "quantconnect")

    secret_values = (settings.webhook_token, settings.qc_api_token)

    def _safe(value: str) -> str:
        return "[redacted]" if value in secret_values else value

    log.info(
        "forwarded pattern=%s signal=%s side=%s ticker=%s",
        _safe(command["pattern"]),
        _safe(command["signal"]),
        _safe(command["side"]),
        _safe(command["ticker"]),
    )
    return JSONResponse(status_code=200, content={"ok": True})
