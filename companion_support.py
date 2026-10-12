"""
companion_support.py - the "extra" features around the voice pipeline:

  1. Memory + mood trend   (remembers past chats, tracks mood over days)
  2. Safety layer          (flags distress; keyword backstop on top of Gemini)
  3. Caregiver alerts      (Telegram and/or email + always a local log)
  4. Robust JSON parsing of Gemini's answer

No heavy libraries here (no torch/whisper), so it can be tested anywhere.

PRIVACY: memory and logs are stored locally in plain JSON. Get consent from the
person (or their guardian) before using this with real people.
"""

import json
import os
import re
import smtplib
import ssl
from datetime import datetime, timedelta
from email.message import EmailMessage

MEMORY_FILE = "companion_memory.json"
ALERT_LOG_FILE = "alerts_log.json"

EMOTIONS = ["happy", "sad", "anxious", "lonely", "scared", "angry", "neutral"]
RISK_LEVELS = ["none", "concern", "high"]

# +1 good ... -1 low. Used for the mood trend.
VALENCE = {
    "happy": 1.0,
    "neutral": 0.0,
    "angry": -0.7,
    "anxious": -1.0,
    "sad": -1.0,
    "lonely": -1.0,
    "scared": -1.0,
}

LOW_DAY_THRESHOLD = -0.4      # a day with average mood at/below this counts as "low"
STREAK_DAYS_FOR_ALERT = 3     # this many low days in a row triggers a caregiver alert
ALERT_COOLDOWN_HOURS = 24     # don't send the same alert reason more than once a day


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------
def _read_json_list(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalise_emotion(value):
    value = str(value or "").lower().strip()
    return value if value in EMOTIONS else "neutral"


def normalise_risk(value):
    value = str(value or "").lower().strip()
    return value if value in RISK_LEVELS else "none"


def parse_model_json(text):
    """Parse Gemini's reply even if it wrapped the JSON in ``` fences or extra text."""
    if not text:
        raise ValueError("Empty response from model")
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ------------------------------------------------------------------
# 1. MEMORY + MOOD TREND
# ------------------------------------------------------------------
def load_memory(path=MEMORY_FILE):
    return _read_json_list(path)


def add_record(record, path=MEMORY_FILE):
    records = load_memory(path)
    records.append(record)
    _write_json(path, records)
    return records


def clear_memory(path=MEMORY_FILE):
    if os.path.exists(path):
        os.remove(path)


def _parse_ts(value):
    return datetime.fromisoformat(value)


def _days_ago_text(ts, now):
    days = (now.date() - ts.date()).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def recent_context(records, n=4, now=None):
    """A short plain-English summary of the last few chats, to give Gemini memory."""
    now = now or datetime.now()
    if not records:
        return "No previous conversations."
    lines = []
    for r in records[-n:]:
        ts = _parse_ts(r["timestamp"])
        said = (r.get("transcript") or "").strip().replace("\n", " ")
        if len(said) > 100:
            said = said[:100] + "..."
        lines.append(f'- {_days_ago_text(ts, now)}: seemed {r.get("emotion", "neutral")}; said: "{said}"')
    return "\n".join(lines)


def mood_summary(records, now=None, window_days=14):
    """
    Returns a dict:
      days_with_data, streak_low_days, trend (improving/declining/stable/not enough data),
      today_avg, last_emotion, emotion_counts
    """
    now = now or datetime.now()
    since = now - timedelta(days=window_days)
    recent = [r for r in records if _parse_ts(r["timestamp"]) >= since]

    counts = {e: 0 for e in EMOTIONS}
    by_day = {}
    for r in recent:
        emo = normalise_emotion(r.get("emotion"))
        counts[emo] += 1
        day = _parse_ts(r["timestamp"]).date()
        by_day.setdefault(day, []).append(VALENCE[emo])

    daily = {d: sum(v) / len(v) for d, v in by_day.items()}
    days_sorted = sorted(daily)

    # consecutive low days ending at the latest day with data (must be today or yesterday)
    streak = 0
    if days_sorted and (now.date() - days_sorted[-1]).days <= 1:
        prev = None
        for d in reversed(days_sorted):
            if daily[d] > LOW_DAY_THRESHOLD:
                break
            if prev is not None and (prev - d).days != 1:
                break
            streak += 1
            prev = d

    trend = "not enough data"
    if len(days_sorted) >= 4:
        last3 = [daily[d] for d in days_sorted[-3:]]
        before = [daily[d] for d in days_sorted[-6:-3]]
        diff = sum(last3) / len(last3) - sum(before) / len(before)
        trend = "improving" if diff > 0.3 else "declining" if diff < -0.3 else "stable"

    return {
        "days_with_data": len(days_sorted),
        "streak_low_days": streak,
        "trend": trend,
        "today_avg": daily.get(now.date()),
        "last_emotion": normalise_emotion(recent[-1].get("emotion")) if recent else None,
        "emotion_counts": counts,
    }


def mood_text(summary):
    """One line for the Gemini prompt."""
    if summary["days_with_data"] == 0:
        return "This is the first time you are talking to them."
    parts = [f"Mood trend over recent days: {summary['trend']}."]
    if summary["streak_low_days"] >= 2:
        parts.append(f"They have seemed low for {summary['streak_low_days']} days in a row, so be extra gentle.")
    return " ".join(parts)


# ------------------------------------------------------------------
# 2. SAFETY LAYER
# ------------------------------------------------------------------
def keyword_risk(transcript, danger_words):
    text = (transcript or "").lower()
    return any(w.lower() in text for w in danger_words)


def final_risk(model_risk, transcript, danger_words):
    """High if EITHER Gemini says high OR a backstop keyword is present."""
    if keyword_risk(transcript, danger_words):
        return "high"
    return normalise_risk(model_risk)


# ------------------------------------------------------------------
# 3. CAREGIVER ALERTS
# ------------------------------------------------------------------
def alerts_to_send(risk, summary, alert_log, now=None):
    """Which alert reasons should fire now (after the 24h cooldown per reason)."""
    now = now or datetime.now()
    wanted = []
    if risk == "high":
        wanted.append("distress")
    if summary["streak_low_days"] >= STREAK_DAYS_FOR_ALERT:
        wanted.append("persistent_low_mood")

    fire = []
    for reason in wanted:
        last = None
        for entry in alert_log:
            if reason in entry.get("reasons", []):
                t = _parse_ts(entry["timestamp"])
                last = t if last is None or t > last else last
        if last is None or now - last >= timedelta(hours=ALERT_COOLDOWN_HOURS):
            fire.append(reason)
    return fire


def build_alert_message(reasons, summary, lang_name, snippet=None):
    lines = [f"Companion bot notice ({lang_name})."]
    if "distress" in reasons:
        lines.append("They said something that may mean they are in distress or danger. Please check on them as soon as you can.")
    if "persistent_low_mood" in reasons:
        lines.append(f"They have seemed low (sad / lonely / anxious) for {summary['streak_low_days']} days in a row. A call or visit may help.")
    if snippet and os.environ.get("ALERT_INCLUDE_TEXT") == "1":
        lines.append(f'What they said: "{snippet}"')
    lines.append("This is an automated notice, not a diagnosis.")
    return "\n".join(lines)


def _send_telegram(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return None
    import requests
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message},
        timeout=10,
    )
    r.raise_for_status()
    return "telegram"


def _send_email(message):
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    to = os.environ.get("ALERT_TO")
    if not (host and user and password and to):
        return None
    msg = EmailMessage()
    msg["Subject"] = "Companion bot notice"
    msg["From"] = user
    msg["To"] = to
    msg.set_content(message)
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=15) as s:
        s.login(user, password)
        s.send_message(msg)
    return "email"


def send_alert(reasons, summary, lang_name, snippet=None, path=ALERT_LOG_FILE, now=None):
    """Send via any configured channel and ALWAYS record it in alerts_log.json."""
    now = now or datetime.now()
    message = build_alert_message(reasons, summary, lang_name, snippet)
    delivered, errors = [], []
    for sender in (_send_telegram, _send_email):
        try:
            channel = sender(message)
            if channel:
                delivered.append(channel)
        except Exception as exc:  # never let an alert failure crash the bot
            errors.append(f"{sender.__name__}: {exc}")

    entry = {
        "timestamp": now.isoformat(timespec="seconds"),
        "reasons": reasons,
        "delivered_via": delivered,
        "errors": errors,
        "note": "logged only (no channel configured)" if not delivered and not errors else "",
        "message": message,
    }
    log = _read_json_list(path)
    log.append(entry)
    _write_json(path, log)
    return entry


def load_alert_log(path=ALERT_LOG_FILE):
    return _read_json_list(path)
