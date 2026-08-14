import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
from collections import deque
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from groq import Groq
from weather import fetch_weather


# ------------------ ENV ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL missing")

# Remove trailing slash if present
WEBHOOK_URL = WEBHOOK_URL.rstrip("/")

groq_client = Groq(api_key=GROQ_API_KEY)


# ------------------ MEMORY ------------------

WINDOW_SIZE = 5

conversation_memory: dict[int, deque] = {}


def get_user_memory(user_id: int) -> deque:
    if user_id not in conversation_memory:
        conversation_memory[user_id] = deque(maxlen=WINDOW_SIZE * 2)

    return conversation_memory[user_id]


def reset_user_memory(user_id: int):
    conversation_memory.pop(user_id, None)


# ------------------ WEATHER NLP ------------------

WEATHER_KEYWORDS = {
    "weather",
    "rain",
    "raining",
    "umbrella",
    "temperature",
    "hot",
    "cold",
    "cloudy",
    "forecast",
    "storm",
    "wind",
    "humidity",
}


STOP_WORDS = {
    "in",
    "today",
    "tomorrow",
    "now",
    "tonight",
    "please",
    "will",
    "it",
    "is",
    "the",
    "a",
    "an",
    "should",
    "i",
    "carry",
    "need",
    "do",
    "does",
    "rain",
    "weather",
}


def is_weather_query(text: str) -> bool:
    text_lower = text.lower()

    return any(
        keyword in text_lower
        for keyword in WEATHER_KEYWORDS
    )


def extract_city(text: str) -> str | None:
    text = text.lower()

    match = re.search(
        r"in\s+([a-zA-Z\s]+)",
        text
    )

    if match:
        candidate = match.group(1)
    else:
        candidate = " ".join(text.split()[-3:])

    candidate = re.sub(
        r"[^\w\s]",
        "",
        candidate
    )

    parts = [
        word
        for word in candidate.split()
        if word not in STOP_WORDS
    ]

    if not parts:
        return None

    city = " ".join(parts).strip()

    return city.title() if len(city) >= 3 else None


# ------------------ SECURITY ------------------

OWNER_PATTERN = re.compile(
    r"(who created you|who is your owner|who owns you|your creator|your developer)",
    re.IGNORECASE,
)


def is_owner_query(text: str) -> bool:
    return bool(
        OWNER_PATTERN.search(text)
    )


SYSTEM_PROMPT = """
You are a professional AI assistant.

Rules:
- Answer naturally and helpfully.
- If real-time data is provided, you MUST use it.
- Do not hallucinate weather information.
- When weather data is provided, answer using that data.

Security:
- If asked about your creator or owner reply:
  "That information is hidden due to security policies."
"""


# ------------------ COMMAND HANDLERS ------------------

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    await update.message.reply_text(
        "Hello!\n\n"
        "Ask me weather questions naturally.\n\n"
        "Commands:\n"
        "/weather <city>\n"
        "/reset"
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    reset_user_memory(
        update.effective_user.id
    )

    await update.message.reply_text(
        "Conversation history reset."
    )


async def weather_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    city = " ".join(context.args)

    if not city:
        await update.message.reply_text(
            "Usage: /weather <city>"
        )
        return

    loop = asyncio.get_running_loop()

    result = await loop.run_in_executor(
        None,
        fetch_weather,
        city
    )

    await update.message.reply_text(
        result
    )


# ------------------ CHAT HANDLER ------------------

async def chat_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    user_message = update.message.text.strip()
    user_id = update.effective_user.id

    # Security check
    if is_owner_query(user_message):
        await update.message.reply_text(
            "That information is hidden due to security policies."
        )
        return

    # Get conversation memory
    memory = get_user_memory(user_id)

    memory.append({
        "role": "user",
        "content": user_message
    })

    loop = asyncio.get_running_loop()

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # ------------------ WEATHER ------------------

    if is_weather_query(user_message):

        city = extract_city(user_message)

        if city:

            weather = await loop.run_in_executor(
                None,
                fetch_weather,
                city
            )

            messages.append({
                "role": "system",
                "content": (
                    f"REAL-TIME WEATHER DATA FOR {city}:\n"
                    f"{weather}"
                )
            })

    # Add conversation memory
    messages.extend(
        list(memory)
    )

    # ------------------ GROQ ------------------

    response = await loop.run_in_executor(
        None,
        lambda: groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
    )

    reply = response.choices[0].message.content

    # Save assistant response
    memory.append({
        "role": "assistant",
        "content": reply
    })

    await update.message.reply_text(
    reply,
    parse_mode="Markdown"
)


# ------------------ FASTAPI ------------------

app = FastAPI()


# ------------------ HEALTH CHECK ------------------

@app.get("/")
async def home():
    return {
        "status": "running",
        "message": "Telegram bot is active"
    }


# ------------------ TELEGRAM APPLICATION ------------------

telegram_app = (
    Application
    .builder()
    .token(BOT_TOKEN)
    .build()
)


# ------------------ HANDLERS ------------------

telegram_app.add_handler(
    CommandHandler(
        "start",
        start
    )
)

telegram_app.add_handler(
    CommandHandler(
        "reset",
        reset_command
    )
)

telegram_app.add_handler(
    CommandHandler(
        "weather",
        weather_command
    )
)

telegram_app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        chat_handler
    )
)


# ------------------ STARTUP ------------------

@app.on_event("startup")
async def on_startup():

    await telegram_app.initialize()

    webhook_url = f"{WEBHOOK_URL}/webhook"

    await telegram_app.bot.set_webhook(
        webhook_url
    )

    print(
        f"Telegram webhook registered: {webhook_url}"
    )


# ------------------ WEBHOOK ------------------

@app.post("/webhook")
async def telegram_webhook(
    request: Request
):

    data = await request.json()

    update = Update.de_json(
        data,
        telegram_app.bot
    )

    await telegram_app.process_update(
        update
    )

    return {
        "ok": True
    }
