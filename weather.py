import sys
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

def fetch_weather(city: str) -> str:
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    r = requests.get(url, params=params, timeout=5)
    r.raise_for_status()
    data = r.json()

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

def main():
    try:
        line = sys.stdin.readline()
        if not line:
            return

        payload = json.loads(line.strip())
        city = payload.get("city")

        if not city:
            print(json.dumps({"result": "City not provided"}), flush=True)
            return

        result = fetch_weather(city)
        print(json.dumps({"result": result}), flush=True)

    except Exception:
        print(json.dumps({"result": "Weather service error"}), flush=True)

if __name__ == "__main__":
    main()
