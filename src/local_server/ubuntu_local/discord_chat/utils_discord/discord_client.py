"""
discord_client.py
Utility: builds the pong response payload for a Discord ping.
"""

import json
from datetime import datetime, timezone


def build_pong_response(channel_name: str) -> str:
    """
    Build a formatted JSON pong reply to send back to Discord.

    Args:
        channel_name: The Discord channel name where ping was received.

    Returns:
        A code-block formatted string ready for channel.send()
    """
    payload = {
        "status": "pong",
        "server": "overlander26-local",
        "received": "ping",
        "channel": channel_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    json_str = json.dumps(payload, indent=2)
    # Wrap in a Discord code block so it renders cleanly in chat
    return f"```json\n{json_str}\n```"
