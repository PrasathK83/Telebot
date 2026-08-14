# AI Weather Telegram Bot

A Telegram chatbot built with Python that provides real-time weather information and AI-powered responses using Groq and OpenWeatherMap APIs.

## Features

* Real-time weather information
* Natural language weather queries
* Telegram bot integration
* AI responses using Groq LLM
* Short-term conversation memory
* Weather command using `/weather <city>`
* Conversation reset using `/reset`
* FastAPI webhook
* Environment variable based API key management

## Technologies Used

* Python
* FastAPI
* python-telegram-bot
* Groq
* OpenWeatherMap API
* Requests
* python-dotenv
* asyncio

## Project Structure

```text
project/
│
├── main.py
├── weather.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Environment Variables

Create a `.env` file:

```env
BOT_TOKEN=your_telegram_bot_token
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
WEBHOOK_URL=https://your-domain.com
```

Do not upload the `.env` file to GitHub.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

For Telegram webhooks, the application needs a publicly accessible HTTPS URL.

Set the URL in `.env`:

```env
WEBHOOK_URL=https://your-domain.com
```

The webhook endpoint will be:

```text
https://your-domain.com/webhook
```

## Bot Commands

### Start

```text
/start
```

Displays the available commands.

### Weather

```text
/weather Chennai
```

Returns the current weather information for the specified city.

### Reset

```text
/reset
```

Clears the user's conversation history.

## Natural Language Queries

The bot can understand messages such as:

```text
What's the weather in Chennai?
```

```text
Will it rain in Coimbatore?
```

```text
What is the temperature in Madurai?
```

The bot detects weather-related queries, extracts the city, retrieves real-time weather data, and provides the information through the AI assistant.

## Weather Information

The weather service retrieves:

* Temperature
* Humidity
* Weather condition
* Rain expectation

## Conversation Memory

The bot stores a limited number of recent messages for each Telegram user.

The `/reset` command can be used to clear the stored conversation history.

The conversation memory is stored in application memory, so it will be cleared when the application restarts.

## API Flow

```text
Telegram User
      |
      v
Telegram Bot
      |
      v
FastAPI Webhook
      |
      v
Message Handler
      |
      +------------------+
      |                  |
      v                  v
Weather Query        Normal Query
      |                  |
      v                  v
OpenWeatherMap          Groq
      |                  |
      +--------+---------+
               |
               v
          Bot Response
               |
               v
         Telegram User
```

## Requirements

Example `requirements.txt`:

```text
fastapi
uvicorn
python-telegram-bot
groq
python-dotenv
requests
```

## Security

API keys and bot tokens are stored in environment variables.

Add the following to `.gitignore`:

```text
.env
.venv/
__pycache__/
*.pyc
```

Never commit API keys, bot tokens, or other secrets to GitHub.

## Future Improvements

* Weather forecast
* Weather alerts
* Telegram location sharing
* Persistent conversation memory
* Multi-language support
* Better city detection
* Weather history
