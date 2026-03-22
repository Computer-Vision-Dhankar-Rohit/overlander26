"""
discord_bot.py
discord.py Bot subclass — listens for "ping" in channels and replies with JSON.
"""

import os
import logging

import discord

from discord_client import build_pong_response

logger = logging.getLogger(__name__)

# Intents: need message_content to read text (must be enabled in Dev Portal too)
intents = discord.Intents.default()
intents.message_content = True


class PingBot(discord.Client):
    """
    Discord bot that responds to 'ping' messages with a JSON pong.

    Optional env var DISCORD_CHANNEL_ID restricts listening to one channel.
    If not set, bot responds in any channel it can see.
    """

    def __init__(self):
        super().__init__(intents=intents)
        # Optional: restrict to a specific channel ID
        raw = os.environ.get("DISCORD_CHANNEL_ID", "").strip()
        self.watch_channel_id: int | None = int(raw) if raw else None

    async def on_ready(self):
        logger.info("Discord bot logged in as %s (id=%s)", self.user, self.user.id)
        if self.watch_channel_id:
            logger.info("Watching channel id: %s", self.watch_channel_id)
        else:
            logger.info("Watching ALL channels (no DISCORD_CHANNEL_ID set)")

    async def on_message(self, message: discord.Message):
        # Ignore messages sent by this bot itself (prevents echo loop)
        if message.author == self.user:
            return

        # Ignore other bots
        if message.author.bot:
            return

        # Channel filter (if configured)
        if self.watch_channel_id and message.channel.id != self.watch_channel_id:
            return

        text = message.content.strip().lower()

        if text == "ping":
            channel_name = getattr(message.channel, "name", str(message.channel.id))
            logger.info(
                "Received ping from %s in #%s", message.author, channel_name
            )
            reply = build_pong_response(channel_name)
            await message.channel.send(reply)
            logger.info("Pong sent to #%s", channel_name)


# Singleton bot instance — imported by app_discord.py
bot = PingBot()
