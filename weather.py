import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise RuntimeError("OPENWEATHER_API_KEY missing in environment")

def fetch_weather(city: str) -> str:
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }

        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        condition = data["weather"][0]["description"].lower()

        will_rain = "rain" in condition or humidity > 75

        return (
            f"City: {city}\n"
            f"Temperature: {temp}°C\n"
            f"Humidity: {humidity}%\n"
            f"Condition: {condition}\n"
            f"Rain expected: {'Yes' if will_rain else 'No'}"
        )

    except requests.exceptions.HTTPError:
        return " City not found. Please check the city name."
    except Exception:
        return "⚠ Weather service temporarily unavailable."
