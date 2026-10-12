"""
test_companion_logic.py - checks memory, mood trend, safety and alert rules.
No models, no internet, no API key needed. Run:   python test_companion_logic.py
Uses a temporary folder, so it never touches your real memory or alerts files.
"""
import os
import tempfile
from datetime import datetime, timedelta

import companion_support as sup
from languages import LANGUAGES, build_captions
import companion

NOW = datetime(2026, 10, 10, 18, 0, 0)


def rec(days_ago, emotion, text="x"):
    ts = (NOW - timedelta(days=days_ago)).replace(hour=10)
    return {"timestamp": ts.isoformat(timespec="seconds"), "emotion": emotion, "transcript": text}


def test_mood_streak_and_alert():
    records = [rec(4, "happy"), rec(3, "neutral"), rec(2, "lonely"), rec(1, "sad"), rec(0, "anxious")]
    s = sup.mood_summary(records, NOW)
    assert s["streak_low_days"] == 3, s
    assert s["trend"] == "declining", s
    assert sup.alerts_to_send("none", s, [], NOW) == ["persistent_low_mood"]


def test_streak_broken_by_gap_or_good_day():
    gap = [rec(3, "sad"), rec(2, "sad"), rec(0, "sad")]           # missing yesterday
    assert sup.mood_summary(gap, NOW)["streak_low_days"] == 1
    good = [rec(2, "sad"), rec(1, "happy"), rec(0, "sad")]
    assert sup.mood_summary(good, NOW)["streak_low_days"] == 1


def test_no_alert_when_fine():
    records = [rec(2, "happy"), rec(1, "neutral"), rec(0, "happy")]
    s = sup.mood_summary(records, NOW)
    assert s["streak_low_days"] == 0
    assert sup.alerts_to_send("none", s, [], NOW) == []


def test_distress_alert_and_cooldown():
    s = sup.mood_summary([rec(0, "sad")], NOW)
    assert sup.alerts_to_send("high", s, [], NOW) == ["distress"]
    log = [{"timestamp": (NOW - timedelta(hours=2)).isoformat(), "reasons": ["distress"]}]
    assert sup.alerts_to_send("high", s, log, NOW) == []                        # within 24h
    old = [{"timestamp": (NOW - timedelta(hours=30)).isoformat(), "reasons": ["distress"]}]
    assert sup.alerts_to_send("high", s, old, NOW) == ["distress"]              # cooldown over


def test_keyword_backstop():
    for code, cfg in LANGUAGES.items():
        word = cfg["danger_words"][0]
        assert sup.final_risk("none", f"blah {word} blah", cfg["danger_words"]) == "high", code
        assert sup.final_risk("none", "hello", cfg["danger_words"]) == "none"
        assert sup.final_risk("high", "hello", cfg["danger_words"]) == "high"
        assert sup.final_risk("weird", "hello", cfg["danger_words"]) == "none"


def test_memory_roundtrip_and_context():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "mem.json")
        assert sup.load_memory(path) == []
        sup.add_record(rec(1, "sad", "I felt alone"), path)
        sup.add_record(rec(0, "happy", "I played"), path)
        records = sup.load_memory(path)
        assert len(records) == 2
        ctx = sup.recent_context(records, now=NOW)
        assert "yesterday" in ctx and "earlier today" in ctx and "I felt alone" in ctx
        sup.clear_memory(path)
        assert sup.load_memory(path) == []
        assert sup.recent_context([], now=NOW) == "No previous conversations."


def test_send_alert_logs_without_channels():
    for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SMTP_HOST", "SMTP_USER", "SMTP_PASS", "ALERT_TO"):
        os.environ.pop(k, None)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "alerts.json")
        s = sup.mood_summary([rec(0, "sad")], NOW)
        entry = sup.send_alert(["distress"], s, "Tamil", snippet="private words", path=path, now=NOW)
        assert entry["delivered_via"] == [] and "logged only" in entry["note"]
        assert "private words" not in entry["message"]            # text excluded by default
        assert len(sup.load_alert_log(path)) == 1


def test_parse_model_json():
    assert sup.parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert sup.parse_model_json('Sure! {"a": 2} done') == {"a": 2}


def test_captions_and_prompt():
    caps = build_captions("hi")
    assert caps["sad"].startswith("Divya") and set(caps) == set(sup.EMOTIONS)
    assert build_captions("ml")["happy"].startswith("Anjali")
    assert build_captions("ta")["neutral"].startswith("Kavitha")
    p = companion.build_prompt(LANGUAGES["ml"], "No previous conversations.", "mood", "gemini")
    assert "Malayalam" in p and "emotion_from_voice" in p and "audio" in p
    p2 = companion.build_prompt(LANGUAGES["ta"], "ctx", "mood", "whisper", "வணக்கம்")
    assert "வணக்கம்" in p2 and "unknown" in p2


def test_language_detection_helpers():
    from languages import pick_language, all_danger_words, language_list_text
    # Whisper often puts Urdu/Marathi near Hindi; we only pick among OUR languages
    assert pick_language({"ur": 0.5, "hi": 0.3, "ta": 0.1}) == ("hi", 0.3)
    assert pick_language({"te": 0.6, "ta": 0.2})[0] == "te"
    assert pick_language({"fr": 0.9}) is None
    assert companion.resolve_language(None, "te") == "te"
    assert companion.resolve_language("auto", "ml") == "ml"
    assert companion.resolve_language("ta", "te") == "ta"          # forced wins
    assert companion.resolve_language(None, "other") is None       # unsupported -> None
    assert companion.resolve_language(None, "") is None
    assert "Telugu (te)" in language_list_text()
    assert len(all_danger_words()) >= 15


def test_auto_prompt_and_telugu_captions():
    p = companion.build_prompt(None, "ctx", "mood", "gemini")
    assert "Telugu (te)" in p and '"other"' in p and "same language" in p
    caps = build_captions("te")
    assert caps["sad"].startswith("A warm, affectionate female speaker")   # friend's tested captions kept
    assert set(caps) == set(sup.EMOTIONS)
    assert "Telugu" in companion.build_prompt(LANGUAGES["te"], "ctx", "mood", "gemini")


def test_sarvam_code_mapping():
    assert companion.to_sarvam_code("ta") == "ta-IN" and companion.to_sarvam_code(None) == "unknown"
    assert companion.from_sarvam_code("hi-IN") == "hi" and companion.from_sarvam_code(None) is None
    assert companion.resolve_language(None, companion.from_sarvam_code("te-IN")) == "te"
    assert companion.resolve_language(None, companion.from_sarvam_code("kok-IN")) is None   # not supported by us
    for code in LANGUAGES:   # every language we support must round-trip through Sarvam's code format
        assert companion.from_sarvam_code(companion.to_sarvam_code(code)) == code


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} checks passed.")
