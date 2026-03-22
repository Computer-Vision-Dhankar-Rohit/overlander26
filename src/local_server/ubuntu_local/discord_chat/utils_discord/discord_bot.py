"""
discord_bot.py
discord.py Bot subclass — passes all messages to the LangChain agent.
"""

import os
import sys
import logging

import discord

# Allow import of langchain_agents_discord from the fast_api sibling directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fast_api"))

from langchain_agents_discord import agent  # noqa: E402

logger = logging.getLogger(__name__)

# Intents: need message_content to read text (must be enabled in Dev Portal too)
intents = discord.Intents.default()
intents.message_content = True


class PingBot(discord.Client):
    """
    Discord bot powered by a LangChain OpenAI agent.
    Responds to all messages in watched channels — including "ping".

    Optional env var DISCORD_CHANNEL_ID restricts listening to one channel.
    If not set, bot responds in any channel it can see.
    """

    def __init__(self):
        super().__init__(intents=intents)
        raw = os.environ.get("DISCORD_CHANNEL_ID", "").strip()
        self.watch_channel_id: int | None = int(raw) if raw else None

    async def on_ready(self):
        logger.info("Discord bot logged in as %s (id=%s)", self.user, self.user.id)
        if self.watch_channel_id:
            logger.info("Watching channel id: %s", self.watch_channel_id)
        else:
            logger.info("Watching ALL channels (no DISCORD_CHANNEL_ID set)")

    async def on_message(self, message: discord.Message):
        # Ignore messages from this bot (prevents echo loop)
        if message.author == self.user:
            return

        # Ignore other bots
        if message.author.bot:
            return

        # Channel filter (if configured)
        if self.watch_channel_id and message.channel.id != self.watch_channel_id:
            return

        text = message.content.strip()
        if not text:
            return

        channel_name = getattr(message.channel, "name", str(message.channel.id))
        author = str(message.author)

        logger.info("Message from %s in #%s: %s", author, channel_name, text[:80])

        # Show typing indicator while agent processes
        async with message.channel.typing():
            reply = await agent.respond(
                message=text,
                channel=channel_name,
                author=author,
            )

        await message.channel.send(reply)


# Singleton bot instance — imported by app_discord.py
bot = PingBot()
