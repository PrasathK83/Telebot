from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()

TOKEN = os.getenv("TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not TOKEN:
    raise ValueError("Telegram BOT TOKEN is missing in .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------- COMMAND HANDLERS ---------------- #

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! I'm TeleBot.\nAsk me anything!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - Start the bot\n"
        "/help - Show help menu\n"
        "/content - About TeleBot\n\n"
        "Just send any message to chat with AI "
    )

async def content(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "TeleBot uses Groq's LLaMA 3.3 70B Versatile model to answer your questions."
    )

# ---------------- AI CHAT HANDLER ---------------- #

async def chat_with_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        ai_reply = response.choices[0].message.content
        await update.message.reply_text(ai_reply)

    except Exception as e:
        await update.message.reply_text("Error while generating response.")
        print("Groq Error:", e)

# ---------------- APP SETUP ---------------- #

app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("help", help_command))
app.add_handler(CommandHandler("content", content))

# Handles ALL normal text messages
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_ai))

# ---------------- RUN BOT ---------------- #

if __name__ == "__main__":
    print("TeleBot is running...")
    app.run_polling()
