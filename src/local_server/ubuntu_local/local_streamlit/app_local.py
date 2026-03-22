"""
app_local.py
FastAPI server — receives WhatsApp pings and replies with JSON.

Run:
    uvicorn app_local:app --host 0.0.0.0 --port 8000 --reload

Required env vars (see .env.example):
    WHATSAPP_VERIFY_TOKEN
    WHATSAPP_PHONE_NUMBER_ID
    WHATSAPP_ACCESS_TOKEN
"""

import os
import json
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from whatsapp_client import send_whatsapp_message

# ── env & logging ──────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Overlander26 WhatsApp Webhook", version="0.1.0")


# ── health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ── webhook verification (Meta calls this once when you register the webhook) ──
@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Meta sends:
        GET /webhook?hub.mode=subscribe&hub.verify_token=<token>&hub.challenge=<random>
    We must echo back hub.challenge if the token matches.
    """
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")

    if mode == "subscribe" and token == verify_token:
        logger.info("Webhook verified by Meta.")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("Webhook verification failed. mode=%s token=%s", mode, token)
    raise HTTPException(status_code=403, detail="Verification failed")


# ── incoming messages ──────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(request: Request):
    """
    Meta POSTs incoming WhatsApp messages here.
    We parse the payload, and if the message body is 'ping' we reply.
    Always return HTTP 200 — Meta will retry indefinitely on non-200.
    """
    try:
        body = await request.json()
    except Exception:
        logger.error("Could not parse request body as JSON.")
        return JSONResponse(content={"status": "ignored"}, status_code=200)

    logger.info("Webhook payload: %s", json.dumps(body, indent=2))

    # Guard: only handle whatsapp_business_account objects
    if body.get("object") != "whatsapp_business_account":
        return JSONResponse(content={"status": "ignored"}, status_code=200)

    # Walk the nested structure
    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])

            for message in messages:
                if message.get("type") != "text":
                    continue  # skip non-text (images, audio, etc.)

                sender = message.get("from", "")
                text_body = message.get("text", {}).get("body", "").strip().lower()

                logger.info("Message from %s: %s", sender, text_body)

                if text_body == "ping":
                    await _handle_ping(sender)

    return JSONResponse(content={"status": "ok"}, status_code=200)


# ── ping handler ───────────────────────────────────────────────────────────────
async def _handle_ping(sender: str):
    reply_data = {
        "status": "pong",
        "server": "overlander26-local",
        "received": "ping",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Format the dict as a readable string for the WhatsApp chat
    reply_text = json.dumps(reply_data, indent=2)
    logger.info("Sending pong to %s", sender)
    await send_whatsapp_message(to=sender, text=reply_text)
