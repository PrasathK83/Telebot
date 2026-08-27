import re
from datetime import datetime, timedelta
from telegram.ext import ContextTypes

# ------------------ CONFIG ------------------

# Cascading offsets — schedule_reminders will only use whichever of these
# actually fit before the event; if none fit (event is under a minute
# away), it falls back to a single reminder right at the event time.
REMINDER_OFFSETS = [
    ("1 day before", timedelta(days=1)),
    ("1 hour before", timedelta(hours=1)),
    ("10 minutes before", timedelta(minutes=10)),
    ("5 minutes before", timedelta(minutes=5)),
    ("1 minute before", timedelta(minutes=1)),
]

EVENT_HINTS = re.compile(
    r"\b(meeting|call|appointment|event|reminder|remind me|schedule|"
    r"registration|deadline|due|interview|class|exam|submission)\b",
    re.IGNORECASE,
)

RELATIVE_PATTERN = re.compile(
    r"\b(?:in|for|after)\s+(\d+)\s*"
    r"(sec(?:ond)?s?|min(?:ute)?s?|hrs?|hours?|days?)\b",
    re.IGNORECASE,
)

GENERIC_REMINDER_WORDS = re.compile(
    r"\b(set (a )?reminder|remind me|reminder)\b", re.IGNORECASE
)

TIME_PATTERN = re.compile(
    r"\bat\s+(\d{1,2})(:(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE
)

TOMORROW_PATTERN = re.compile(r"\btomorrow\b", re.IGNORECASE)


def init_scheduler():
    """No-op — JobQueue (built into python-telegram-bot) handles scheduling."""
    pass


# ------------------ EVENT EXTRACTION ------------------

def extract_event(text: str):
    """
    Looks for a schedulable event in free text (chat message or OCR'd
    image text). Returns (description, event_time) or None.
    Handles:
      - relative: "in 10 sec", "in 5 min", "in 2 hours", "in 1 day"
      - absolute: "tomorrow at 5pm", "at 3:30 pm"
    """
    if not EVENT_HINTS.search(text):
        return None

    now = datetime.now()

    # ---- Relative: "in X sec/min/hr/day" ----
        # ---- Relative: "in/for/after X sec/min/hr/day" ----
    match = RELATIVE_PATTERN.search(text)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()

        if unit.startswith("sec"):
            delta = timedelta(seconds=amount)
        elif unit.startswith("min"):
            delta = timedelta(minutes=amount)
        elif unit.startswith(("hr", "hour")):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)

        event_time = now + delta

        # Strip the matched time phrase
        description = (text[:match.start()] + text[match.end():]).strip(" -:,.")
        # Strip generic filler like "set reminder" / "remind me" so a bare
        # "set reminder for 2 min" doesn't leave junk as the description
        description = GENERIC_REMINDER_WORDS.sub("", description).strip(" -:,.")

        return (description or "your reminder"), event_time

    # ---- Absolute: "[tomorrow] at H(:MM)(am/pm)" ----
    time_match = TIME_PATTERN.search(text)
    if time_match:
        day_match = TOMORROW_PATTERN.search(text)
        day_offset = 1 if day_match else 0

        hour = int(time_match.group(1))
        minute = int(time_match.group(3)) if time_match.group(3) else 0
        meridian = (time_match.group(4) or "").lower()

        if meridian == "pm" and hour != 12:
            hour += 12
        if meridian == "am" and hour == 12:
            hour = 0

        event_time = (now + timedelta(days=day_offset)).replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if event_time <= now:
            event_time += timedelta(days=1)

        description = text
        if day_match:
            description = description.replace(day_match.group(0), "")
        description = description.replace(time_match.group(0), "")
        description = description.strip(" -:,.")

        return (description or "your event"), event_time

    return None


# ------------------ REMINDER CALLBACK ------------------

async def send_reminder(context: ContextTypes.DEFAULT_TYPE):
    data = context.job.data
    await context.bot.send_message(
        chat_id=context.job.chat_id,
        text=(
            f"🔔 Reminder ({data['label']}): \"{data['description']}\"\n"
            f"Happening at {data['event_time'].strftime('%a %b %d, %I:%M %p')}"
        ),
    )


# ------------------ SCHEDULING ------------------
def parse_reminder_command(args_text: str):
    """
    Parses the raw argument string from /reminder command, e.g.
    '20 sec submit form' or '2 min' or '1 hour team call'.
    Returns (description, event_time) or None if it can't parse a time.
    """
    match = re.match(
        r"\s*(\d+)\s*(sec(?:ond)?s?|min(?:ute)?s?|hrs?|hours?|days?)\s*(.*)",
        args_text.strip(),
        re.IGNORECASE,
    )
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2).lower()
    description = match.group(3).strip() or "your reminder"

    if unit.startswith("sec"):
        delta = timedelta(seconds=amount)
    elif unit.startswith("min"):
        delta = timedelta(minutes=amount)
    elif unit.startswith(("hr", "hour")):
        delta = timedelta(hours=amount)
    else:
        delta = timedelta(days=amount)

    event_time = datetime.now() + delta
    return description, event_time

def schedule_reminders(job_queue, chat_id: int, description: str, event_time: datetime):
    """
    Schedules one job per offset that still fits before event_time.
    If NONE of the offsets fit (event is under a minute away), schedules
    a single reminder right at event_time instead.
    Returns the list of labels actually scheduled.
    """
    now = datetime.now()
    labels_set = []

    for label, offset in REMINDER_OFFSETS:
        due_seconds = (event_time - offset - now).total_seconds()
        if due_seconds > 0:
            job_queue.run_once(
                send_reminder,
                due_seconds,
                chat_id=chat_id,
                name=f"{chat_id}-{description}-{label}",
                data={"description": description, "label": label, "event_time": event_time},
            )
            labels_set.append(label)

    if not labels_set:
        due_seconds = (event_time - now).total_seconds()
        if due_seconds > 0:
            job_queue.run_once(
                send_reminder,
                due_seconds,
                chat_id=chat_id,
                name=f"{chat_id}-{description}-at-event",
                data={"description": description, "label": "now", "event_time": event_time},
            )
            labels_set.append("right at event time")

    return labels_set
