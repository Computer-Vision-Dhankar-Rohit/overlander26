"""
whatsapp_client.py
Helper for sending WhatsApp messages via Meta Cloud API.
"""

import os
import httpx
import logging

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v18.0"


async def send_whatsapp_message(to: str, text: str) -> dict:
    """
    Send a text message to a WhatsApp number via Meta Cloud API.

    Args:
        to:   Recipient phone number in international format (e.g. "919876543210")
        text: Message body to send

    Returns:
        Parsed JSON response from Meta API
    """
    phone_number_id = os.environ["WHATSAPP_PHONE_NUMBER_ID"]
    access_token = os.environ["WHATSAPP_ACCESS_TOKEN"]

    url = f"{GRAPH_API_URL}/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text},
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        logger.error("Meta API error %s: %s", response.status_code, response.text)
    else:
        logger.info("Message sent to %s", to)

    return response.json()
