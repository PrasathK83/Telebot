import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import re
import tempfile
from collections import deque
from dotenv import load_dotenv

from fastapi import FastAPI, Request
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

from PIL import Image, ImageOps, ImageEnhance
import easyocr
import numpy as np


# ------------------ ENV ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

if BOT_TOKEN:
    BOT_TOKEN = BOT_TOKEN.strip().strip('"').strip("'")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip().strip('"').strip("'")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing")
if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL missing")

# Remove trailing slash if present
WEBHOOK_URL = WEBHOOK_URL.rstrip("/")

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=20.0,
    max_retries=1,
)


# ------------------ OCR SETUP ------------------
# EasyOCR model loading is slow (downloads + loads torch models). Doing this
# at import time blocks uvicorn from binding the port, which makes Render's
# port scan time out. So we lazy-load it on first use instead, inside a
# thread executor.

ocr_reader = None


def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        print("Loading EasyOCR (first use)...")
        ocr_reader = easyocr.Reader(["en"], gpu=False)
        print("EasyOCR loaded successfully.")
    return ocr_reader


# ------------------ MEMORY ------------------

WINDOW_SIZE = 5
conversation_memory: dict[int, deque] = {}

# Holds OCR text extracted from a user's last image, waiting for their
# ONE follow-up question about it. Cleared after that single answer.
pending_ocr: dict[int, str] = {}


def get_user_memory(user_id: int) -> deque:
    if user_id not in conversation_memory:
        conversation_memory[user_id] = deque(maxlen=WINDOW_SIZE * 2)
    return conversation_memory[user_id]


def reset_user_memory(user_id: int):
    conversation_memory.pop(user_id, None)
    pending_ocr.pop(user_id, None)


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
- When weather data is provided, answer using that data.
- Use Markdown formatting when it improves readability.
- Do not unnecessarily make answers very long.

Security:
- If asked about your creator or owner reply:
  "That information is hidden due to security policies."
"""

OCR_SYSTEM_PROMPT = """
You are a professional AI assistant answering a question about text that
was extracted (via OCR) from an image the user just sent.

Rules:
- The OCR text may contain errors, missing spaces, or misreads. Use your
  best judgement to interpret it.
- Answer ONLY the user's specific question about this image content.
- Be concise and directly address what was asked.
- If the OCR text does not contain enough information to answer, say so
  honestly instead of making things up.

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


# ------------------ OCR PIPELINE ------------------

def prepare_ocr_images(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_width, original_height = image.size
    max_dimension = 1800
    largest_dimension = max(original_width, original_height)

    if largest_dimension > max_dimension:
        scale = max_dimension / largest_dimension
        new_size = (
            max(1, int(original_width * scale)),
            max(1, int(original_height * scale)),
        )
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    else:
        min_dimension = 1100
        if largest_dimension < min_dimension:
            scale = min_dimension / largest_dimension
            new_size = (
                max(1, int(original_width * scale)),
                max(1, int(original_height * scale)),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)

    original = image.copy()
    gray = ImageOps.grayscale(image)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)
    enhanced = gray.convert("RGB")

    return [("original", original), ("enhanced", enhanced)]


def run_easyocr(image: Image.Image):
    image_array = np.array(image)
    reader = get_ocr_reader()
    return reader.readtext(
        image_array,
        detail=1,
        paragraph=False,
        canvas_size=1800,
        mag_ratio=1.2,
        text_threshold=0.50,
        low_text=0.20,
        link_threshold=0.20,
        width_ths=0.7,
        height_ths=0.7,
    )


def normalize_ocr_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def extract_detections(results):
    detections = []

    for detection in results:
        if len(detection) < 3:
            continue

        text = str(detection[1]).strip()

        try:
            confidence = float(detection[2])
        except (TypeError, ValueError):
            continue

        if confidence < 0.20 or not text:
            continue

        position = detection[0]
        min_x = min(point[0] for point in position)
        min_y = min(point[1] for point in position)

        detections.append({
            "text": text,
            "confidence": confidence,
            "x": min_x,
            "y": min_y,
            "normalized": normalize_ocr_text(text),
        })

    return detections


def deduplicate_detections(detections):
    unique = {}

    for item in detections:
        key = item["normalized"]
        if not key:
            continue
        if key not in unique or item["confidence"] > unique[key]["confidence"]:
            unique[key] = item

    return list(unique.values())


def extract_text_from_image(image_path: str) -> str:
    print("Starting OCR...")

    image = Image.open(image_path)
    prepared_images = prepare_ocr_images(image)
    all_detections = []

    for name, ocr_image in prepared_images:
        try:
            results = run_easyocr(ocr_image)
            detections = extract_detections(results)
            print(f"OCR pass '{name}': {len(detections)} regions")
            all_detections.extend(detections)
        except Exception as exc:
            print(f"OCR pass '{name}' failed:", exc)

    if not all_detections:
        return ""

    unique_detections = deduplicate_detections(all_detections)
    unique_detections.sort(key=lambda item: (item["y"], item["x"]))

    return "\n".join(item["text"] for item in unique_detections).strip()


# ------------------ COMMAND HANDLERS ------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello!\n\n"
        "I can chat, answer weather questions, and read text from images.\n\n"
        "You can send me:\n"
        "• Text messages (general chat)\n"
        "• Weather questions (e.g. \"will it rain in Chennai today?\")\n"
        "• An image with text — send it, then ask ONE question about it "
        "(e.g. \"summarize this\" or \"what does this say?\"). "
        "I'll only answer about that image once, so make your question count!\n\n"
        "Commands:\n"
        "/weather <city>\n"
        "/reset"
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reset_user_memory(update.effective_user.id)
    await update.message.reply_text("Conversation history reset.")


async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    city = " ".join(context.args)

    if not city:
        await update.message.reply_text("Usage: /weather <city>")
        return

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, fetch_weather, city)
    await update.message.reply_text(result)


# ------------------ IMAGE HANDLER (OCR intake) ------------------

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    processing_message = await update.message.reply_text(
        "Reading text from your image..."
    )

    temp_path = None

    try:
        photo = update.message.photo[-1]
        telegram_file = await context.bot.get_file(photo.file_id)

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        await telegram_file.download_to_drive(temp_path)

        loop = asyncio.get_running_loop()
        extracted_text = await loop.run_in_executor(
            None, extract_text_from_image, temp_path
        )

        if not extracted_text:
            await processing_message.edit_text(
                "I couldn't detect any text in this image. "
                "Try sending a clearer, higher-resolution image."
            )
            return

        # Store for exactly one follow-up question, overwriting any
        # previous pending image for this user.
        pending_ocr[user_id] = extracted_text

        await processing_message.edit_text(
            "Got it — I've read the text from your image.\n\n"
            "What would you like to know about it? "
            "(This applies to this image only, and I'll answer just once — "
            "after that you'll need to send the image again for another question.)"
        )

    except Exception as exc:
        print("Image processing failed:", exc)
        try:
            await processing_message.edit_text(
                "Sorry, I couldn't process this image. "
                "Please try again with a clearer or higher-resolution image."
            )
        except Exception:
            pass

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ------------------ CHAT HANDLER ------------------

async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text.strip()

    if not user_message:
        return

    user_id = update.effective_user.id

    if is_owner_query(user_message):
        await update.message.reply_text(
            "That information is hidden due to security policies."
        )
        return

    loop = asyncio.get_running_loop()

    # ---------- ONE-TIME OCR FOLLOW-UP ----------
    if user_id in pending_ocr:
        ocr_text = pending_ocr.pop(user_id)  # consumed — one-time only

        messages = [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"OCR-extracted text from the image:\n"
                    f"---\n{ocr_text}\n---\n\n"
                    f"My question about this image: {user_message}"
                ),
            },
        ]

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

        except Exception as exc:
            print("OCR-answer Groq call failed:", exc)
            reply = "Sorry, I couldn't process your question about that image."

        await safe_reply(update.message, reply)
        return  # do not fall through to normal chat/weather logic

    # ---------- NORMAL CHAT / WEATHER ----------
    memory = get_user_memory(user_id)
    memory.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if is_weather_query(user_message):
        city = extract_city(user_message)

        if city:
            weather = await loop.run_in_executor(None, fetch_weather, city)
            messages.append({
                "role": "system",
                "content": f"REAL-TIME WEATHER DATA FOR {city}:\n{weather}",
            })

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


# ------------------ FASTAPI ------------------

app = FastAPI()


@app.get("/")
async def home():
    return {
        "status": "running",
        "message": "Telegram bot is active",
    }


# ------------------ TELEGRAM APPLICATION ------------------

telegram_app = Application.builder().token(BOT_TOKEN).build()

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(CommandHandler("reset", reset_command))
telegram_app.add_handler(CommandHandler("weather", weather_command))
telegram_app.add_handler(MessageHandler(filters.PHOTO, image_handler))
telegram_app.add_handler(
    MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler)
)
telegram_app.add_error_handler(error_handler)


# ------------------ STARTUP / SHUTDOWN ------------------

@app.on_event("startup")
async def on_startup():
    await telegram_app.initialize()
    await telegram_app.start()

    webhook_url = f"{WEBHOOK_URL}/webhook"
    await telegram_app.bot.set_webhook(webhook_url)

    print(f"Telegram webhook registered: {webhook_url}")


@app.on_event("shutdown")
async def on_shutdown():
    await telegram_app.stop()
    await telegram_app.shutdown()


# ------------------ WEBHOOK ------------------

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"ok": True}


# ------------------ LOCAL / RENDER ENTRYPOINT ------------------
# On Render, set the start command to:
#   uvicorn bot_webhook:app --host 0.0.0.0 --port $PORT
# This __main__ block is only a convenience for running locally.

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
