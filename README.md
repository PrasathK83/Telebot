TeleBot – AI Telegram Bot using Groq (LLaMA 3)

TeleBot is an AI-powered Telegram chatbot built using python-telegram-bot and Groq’s LLaMA 3.3 70B Versatile model.
It allows users to interact with an AI assistant directly through Telegram using simple text messages.

Features

AI-powered chat using Groq’s LLaMA 3.3 70B model

Asynchronous Telegram bot using python-telegram-bot v20+

Command-based interaction (/start, /help, /content)

Secure configuration using environment variables

Suitable for cloud deployment (Render, Railway, VPS, etc.)

Tech Stack

Python 3.11

python-telegram-bot 20.7

Groq Python SDK

python-dotenv


Prerequisites

Python 3.10 or 3.11

A Telegram Bot Token (from BotFather)

A Groq API Key

Environment Variables

Create a .env file in the project root and add:

TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key


Do not commit the .env file to GitHub.

Installation

clone the repository
cd Telebot


Create a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt

Running the Bot Locally

Start the bot with:

python bot.py


If everything is configured correctly, you should see:

TeleBot is running...

Available Commands

/start – Start the bot

/help – Show help information

/content – Information about the AI model

Any text message – Chat with the AI
