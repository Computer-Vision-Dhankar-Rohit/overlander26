"""
app_discord.py
FastAPI server — manages Discord bot lifecycle and exposes status endpoints.

Run:
    uvicorn app_discord:app --host 0.0.0.0 --port 8001 --reload

Required env vars (see .env.example):
    DISCORD_BOT_TOKEN

Optional env vars:
    DISCORD_CHANNEL_ID   restrict bot to one channel (leave blank for all)
    PORT                 default 8001
"""

import os
import sys
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Allow imports from utils_discord (sibling directory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils_discord"))

from discord_bot import bot  # noqa: E402  (after sys.path patch)

# ── env & logging ──────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── lifespan: start / stop Discord bot with the FastAPI process ────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.error("DISCORD_BOT_TOKEN is not set — bot will NOT connect.")
        yield
        return

    logger.info("Starting Discord bot...")
    bot_task = asyncio.create_task(bot.start(token))

    yield  # server is running

    logger.info("Shutting down Discord bot...")
    await bot.close()
    bot_task.cancel()
    try:
        await bot_task
    except asyncio.CancelledError:
        pass


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Overlander26 Discord Bot Server",
    version="0.1.0",
    lifespan=lifespan,
)


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── discord bot status ─────────────────────────────────────────────────────────
@app.get("/discord/status")
async def discord_status():
    """Returns current Discord bot connection state."""
    is_ready = not bot.is_closed() and bot.is_ready()
    return JSONResponse(
        content={
            "bot_connected": is_ready,
            "bot_user": str(bot.user) if is_ready else None,
            "bot_id": str(bot.user.id) if is_ready else None,
            "watching_channel_id": bot.watch_channel_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
