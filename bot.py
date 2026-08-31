import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
from collections import deque
from dotenv import load_dotenv
from reminders import init_scheduler, extract_event, schedule_reminders, parse_reminder_command
from telegram import Update
from telegram.constants import ParseMode
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
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if BOT_TOKEN:
    BOT_TOKEN = BOT_TOKEN.strip().strip('"').strip("'")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip().strip('"').strip("'")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=20.0,
    max_retries=1,
)


# ------------------ MEMORY ------------------

WINDOW_SIZE = 5
conversation_memory: dict[int, deque] = {}

# Holds an event awaiting yes/no confirmation before reminders are set.
# Auto-confirms if the user doesn't reply within CONFIRM_TIMEOUT seconds.
pending_confirmations: dict[int, dict] = {}
CONFIRM_TIMEOUT = 15  # seconds

CONFIRM_YES = re.compile(r"^\s*(yes|yeah|yep|sure|ok(ay)?|confirm|y)\s*$", re.IGNORECASE)
CONFIRM_NO = re.compile(r"^\s*(no|nope|cancel|n)\s*$", re.IGNORECASE)


def get_user_memory(user_id: int) -> deque:
    if user_id not in conversation_memory:
        conversation_memory[user_id] = deque(maxlen=WINDOW_SIZE * 2)
    return conversation_memory[user_id]


def reset_user_memory(user_id: int):
    conversation_memory.pop(user_id, None)
    pending_confirmations.pop(user_id, None)


# ------------------ WEATHER NLP ------------------

WEATHER_KEYWORDS = {
    "weather", "rain", "raining", "umbrella", "temperature",
    "hot", "cold", "cloudy", "forecast", "storm", "wind", "humidity",
}

STOP_WORDS = {
    "in", "today", "tomorrow", "now", "tonight", "please", "will",
    "it", "is", "the", "a", "an", "should", "i", "carry", "need",
    "do", "does", "rain", "weather",
}


def is_weather_query(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in WEATHER_KEYWORDS)


def extract_city(text: str) -> str | None:
    text = text.lower()

    match = re.search(r"in\s+([a-zA-Z\s]+)", text)

    if match:
        candidate = match.group(1)
    else:
        candidate = " ".join(text.split()[-3:])

    candidate = re.sub(r"[^\w\s]", "", candidate)

    parts = [word for word in candidate.split() if word not in STOP_WORDS]

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
    return bool(OWNER_PATTERN.search(text))


SYSTEM_PROMPT = """
You are a professional AI assistant.

Rules:
- Answer naturally and helpfully.
- If real-time data is provided, you MUST use it.
- Do not hallucinate weather information.
- You CAN set reminders automatically when the user mentions a date/time
  for a meeting or event — this happens outside of you, so just
  acknowledge it naturally if the user asks about it.
- When weather data is provided, answer using that data.
- Use Markdown formatting when it improves readability.
- Do not unnecessarily make answers very long.

Security:
- If asked about your creator or owner reply:
  "That information is hidden due to security policies."
"""


# ------------------ MARKDOWN SAFETY ------------------

def convert_markdown_for_telegram(text: str) -> str:
    lines = text.splitlines()
    result = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("### "):
            result.append(f"*{stripped[4:].strip()}*")
        elif stripped.startswith("## "):
            result.append(f"*{stripped[3:].strip()}*")
        elif stripped.startswith("# "):
            result.append(f"*{stripped[2:].strip()}*")
        else:
            result.append(line)

    return "\n".join(result)


async def safe_reply(message, text: str):
    try:
        await message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception as markdown_error:
        print("Markdown parsing failed:", markdown_error)
        await message.reply_text(text)


# ------------------ REMINDER CONFIRMATION FLOW ------------------

async def _auto_confirm_job(context: ContextTypes.DEFAULT_TYPE):
    """Fired by JobQueue after CONFIRM_TIMEOUT seconds if the user hasn't
    replied yes/no. Confirms and schedules the reminders automatically."""
    user_id = context.job.data["user_id"]
    pending = pending_confirmations.get(user_id)

    # Only proceed if this is still the same pending request (not already
    # answered or replaced by a newer one).
    if not pending or pending.get("token") != context.job.data["token"]:
        return

    pending_confirmations.pop(user_id, None)
    await _finalize_reminder(context, user_id, pending, auto=True)


async def _finalize_reminder(context: ContextTypes.DEFAULT_TYPE, user_id: int, pending: dict, auto: bool):
    labels = schedule_reminders(
        context.job_queue, pending["chat_id"], pending["description"], pending["event_time"]
    )
    prefix = "⏱️ No response, so I went ahead and " if auto else "✅ "
    if labels:
        await context.bot.send_message(
            chat_id=pending["chat_id"],
            text=(
                f"{prefix}set reminders for \"{pending['description']}\" "
                f"({pending['event_time'].strftime('%a %b %d, %I:%M %p')}): "
                f"{', '.join(labels)}."
            ),
        )
    else:
        await context.bot.send_message(
            chat_id=pending["chat_id"],
            text=(
                f"I found \"{pending['description']}\" at "
                f"{pending['event_time'].strftime('%a %b %d, %I:%M %p')}, but it's "
                f"already too late to set any reminder for it."
            ),
        )


async def offer_reminder(context: ContextTypes.DEFAULT_TYPE, chat_id: int, user_id: int, description: str, event_time):
    """Stores a pending confirmation, asks the user, and schedules an
    auto-confirm fallback after CONFIRM_TIMEOUT seconds."""
    token = f"{user_id}-{event_time.timestamp()}"
    pending_confirmations[user_id] = {
        "description": description,
        "event_time": event_time,
        "chat_id": chat_id,
        "token": token,
    }

    context.job_queue.run_once(
        _auto_confirm_job,
        CONFIRM_TIMEOUT,
        data={"user_id": user_id, "token": token},
        name=f"autoconfirm-{token}",
    )

    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            f"📌 I noticed \"{description}\" at "
            f"{event_time.strftime('%a %b %d, %I:%M %p')}. "
            f"Set reminders for it? (yes/no — I'll set them automatically "
            f"in {CONFIRM_TIMEOUT}s if you don't reply)"
        ),
    )


# ------------------ COMMAND HANDLERS ------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello!\n\n"
        "I can chat, answer weather questions, and set reminders for "
        "meetings/events you mention.\n\n"
        "You can send me:\n"
        "• Text messages (general chat)\n"
        "• Weather questions (e.g. \"will it rain in Chennai today?\")\n"
        "• Something with a date/time in it (e.g. \"meeting tomorrow at 5pm\" "
        "or \"registration in 10 min\") — I'll ask to confirm, then set "
        "cascading reminders (1 day / 1 hr / 10 min / 5 min / 1 min before, "
        "whichever fit).\n\n"
        "Commands:\n"
        "/weather <city>\n"
        "/reminder <time> <description> — e.g. /reminder 20 sec check oven\n"
        "/reset"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_user_memory(update.effective_user.id)
    await update.message.reply_text("Conversation history reset.")

async def reminder_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args_text = " ".join(context.args)

    if not args_text:
        await update.message.reply_text(
            "Usage: /reminder <time> <description>\n"
            "e.g. /reminder 20 sec check oven\n"
            "     /reminder 2 min submit form\n"
            "     /reminder 1 hour team call"
        )
        return

    result = parse_reminder_command(args_text)
    if not result:
        await update.message.reply_text(
            "Couldn't understand that. Usage: /reminder <time> <description>\n"
            "e.g. /reminder 20 sec check oven"
        )
        return

    description, event_time = result
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    await offer_reminder(context, chat_id, user_id, description, event_time)
    
async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    city = " ".join(context.args)

    if not city:
        await update.message.reply_text("Usage: /weather <city>")
        return

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, fetch_weather, city)
    await update.message.reply_text(result)


# ------------------ UNSUPPORTED MEDIA HANDLER ------------------

UNSUPPORTED_MEDIA_MESSAGE = (
    "I'm not able to look at images, videos, or voice messages right now — "
    "I can only work with text. Feel free to type out what you'd like help with!"
)


async def unsupported_media_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles photos, videos, voice notes, and other media the bot can't
    process. Replies with a gentle, non-alarming message so the user isn't
    left confused or made to feel at fault."""
    await update.message.reply_text(UNSUPPORTED_MEDIA_MESSAGE)


# ------------------ CHAT HANDLER ------------------

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()

    if not user_message:
        return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if is_owner_query(user_message):
        await update.message.reply_text(
            "That information is hidden due to security policies."
        )
        return

    loop = asyncio.get_running_loop()

    # ---------- REMINDER CONFIRMATION REPLY ----------
    if user_id in pending_confirmations:
        if CONFIRM_YES.match(user_message):
            pending = pending_confirmations.pop(user_id)
            await _finalize_reminder(context, user_id, pending, auto=False)
            return
        if CONFIRM_NO.match(user_message):
            pending_confirmations.pop(user_id, None)
            await update.message.reply_text("Okay, I won't set a reminder for that.")
            return
        # Any other message: leave the pending confirmation active (it
        # will still auto-confirm on its own timer) and fall through so
        # the message gets handled normally below.

    # ---------- SCHEDULE / REMINDER DETECTION ----------
    # Checked before weather, so "meeting tomorrow at 5" isn't hijacked by
    # a stray "tomorrow" match in the weather-keyword check.
    event = extract_event(user_message)
    if event:
        description, event_time = event
        await offer_reminder(context, chat_id, user_id, description, event_time)
        # Falls through so the message still gets a normal chat/weather
        # reply too. Add a `return` here if you'd rather ONLY ask about
        # the reminder and skip the normal AI reply for schedule messages.

    # ---------- NORMAL CHAT / WEATHER ----------
    memory = get_user_memory(user_id)
    memory.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if is_weather_query(user_message):
        city = extract_city(user_message)
        print(f"[weather-nlp] extracted city: {city!r} from: {user_message!r}")

        if city:
            try:
                weather = await loop.run_in_executor(None, fetch_weather, city)
                messages.append({
                    "role": "system",
                    "content": f"REAL-TIME WEATHER DATA FOR {city}:\n{weather}",
                    })
            except Exception as exc:
                print(f"[weather-nlp] fetch_weather failed for {city!r}:", exc)
            # continue without weather data rather than failing the whole reply
    messages.extend(list(memory))

    try:
        response = await loop.run_in_executor(
                None,
                lambda: groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    temperature=0.4,
                    max_completion_tokens=1024,
                    top_p=0.95,
                    stream=False,
                ),
            )

        reply = response.choices[0].message.content or (
            "Sorry, I couldn't generate a response."
        )
        reply = convert_markdown_for_telegram(reply)

        memory.append({"role": "assistant", "content": reply})

    except Exception as exc:
        print("Groq call failed:", exc)

        if memory and memory[-1]["role"] == "user":
            memory.pop()

        reply = "Sorry, I couldn't process your question right now."

    await safe_reply(update.message, reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    print("Telegram error:", context.error)


# ------------------ WEBHOOK SETUP (FastAPI + Render) ------------------
#
# Render is a web service, so it expects something listening on $PORT.
# Instead of run_polling(), we run the bot as a FastAPI app and register
# a webhook with Telegram that points at this service's public URL.
#
# Required env vars:
#   BOT_TOKEN     - already required above
#   WEBHOOK_URL   - your Render service's public base URL,
#                   e.g. https://your-app.onrender.com
#                   (no trailing slash)
#   WEBHOOK_SECRET (optional) - a random string used to validate that
#                   incoming requests really came from Telegram.

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
if WEBHOOK_URL:
    WEBHOOK_URL = WEBHOOK_URL.strip().strip('"').strip("'").rstrip("/")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

# Path Telegram will POST updates to. Includes the bot token so random
# scanners hitting your domain can't spoof updates.
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN}"


telegram_app = Application.builder().token(BOT_TOKEN).build()

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("reset", reset_command))
telegram_app.add_handler(CommandHandler("weather", weather_command))
telegram_app.add_handler(
    MessageHandler(
        filters.PHOTO | filters.VIDEO | filters.VOICE | filters.AUDIO
        | filters.VIDEO_NOTE | filters.ANIMATION | filters.Document.ALL,
        unsupported_media_handler,
    )
)
telegram_app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler)
)
telegram_app.add_handler(CommandHandler("reminder", reminder_command))
telegram_app.add_error_handler(error_handler)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_scheduler()

    await telegram_app.initialize()
    await telegram_app.start()

    if WEBHOOK_URL:
        webhook_kwargs = {
            "url": f"{WEBHOOK_URL}{WEBHOOK_PATH}",
            "drop_pending_updates": True,
            "allowed_updates": Update.ALL_TYPES,
        }
        if WEBHOOK_SECRET:
            webhook_kwargs["secret_token"] = WEBHOOK_SECRET

        await telegram_app.bot.set_webhook(**webhook_kwargs)
        print(f"Webhook set to {WEBHOOK_URL}{WEBHOOK_PATH}")
    else:
        print(
            "WARNING: WEBHOOK_URL is not set — Telegram won't be able to "
            "deliver updates. Set WEBHOOK_URL to your Render service's "
            "public URL."
        )

    yield

    await telegram_app.bot.delete_webhook()
    await telegram_app.stop()
    await telegram_app.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def health_check():
    """Simple endpoint so Render's port scan / uptime checks succeed."""
    return {"status": "ok"}


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if WEBHOOK_SECRET:
        header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if header_secret != WEBHOOK_SECRET:
            return Response(status_code=401)

    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return Response(status_code=200)
