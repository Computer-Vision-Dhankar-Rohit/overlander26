"""
langchain_agents_discord.py
LangChain OpenAI agent for the Discord bot.

Handles all incoming Discord messages and generates intelligent replies
using OpenAI via LangChain. Responds to "ping" and any other message.

Required env var:
    OPENAI_API_KEY   — set in .env file
"""

import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are overlander_1, a helpful assistant bot in the
"Overlander tech's server" Discord server. You are running on a local
Ubuntu server with a static IP.

Rules:
- Keep replies concise and friendly.
- If someone says "ping", reply with a JSON-style status pong that includes
  status, server name ("overlander26-local"), and a timestamp.
- For all other messages, answer helpfully and briefly.
- Never reveal API keys, tokens, or secrets.
- You are built with FastAPI + discord.py + LangChain + OpenAI.
"""


class DiscordLangChainAgent:
    """
    LangChain-powered agent that responds to Discord messages.

    Usage:
        agent = DiscordLangChainAgent()
        reply = await agent.respond("ping", channel="general", author="dhankar")
    """

    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        logger.info("DiscordLangChainAgent initialised with model=%s", model)

    async def respond(self, message: str, channel: str, author: str) -> str:
        """
        Generate a reply for a Discord message.

        Args:
            message: The text the user sent.
            channel: Discord channel name (for context).
            author:  Discord username (for context).

        Returns:
            Reply string to send back to the channel.
        """
        user_content = (
            f"[Channel: #{channel}] [User: {author}]\n{message}"
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            reply = response.content.strip()
            logger.info("Agent reply to %s in #%s: %s", author, channel, reply[:80])
            return reply
        except Exception as e:
            logger.error("LangChain agent error: %s", e)
            return "Sorry, I encountered an error processing your message."


# Singleton — imported by discord_bot.py
agent = DiscordLangChainAgent()
