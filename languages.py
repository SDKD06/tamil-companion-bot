"""
languages.py - per-language settings for the companion bot.

To add a new language, add one entry to LANGUAGES (Whisper code, Indic Parler-TTS
speaker name, pronoun). Nothing else in the code needs to change.

Speakers below are from the Indic Parler-TTS model card
(Tamil: Kavitha / Jaya, Hindi: Divya / Rohit, Malayalam: Anjali / Harish,
English: Mary, Bengali: Aditi, Marathi: Sunita). English/Bengali/Marathi speakers are untested:
listen and swap the speaker name if one sounds wrong.
"""

# Spoken-voice style per emotion. {spk} = speaker name, {she} = she/he
CAPTION_TEMPLATES = {
    "happy": "{spk} speaks in a warm, cheerful, gentle tone, as if sharing good news with someone {she} loves. Very clear audio, close-sounding, no background noise.",
    "sad": "{spk} speaks in a soft, comforting, warm tone, as if consoling someone {she} loves. Very clear audio, close-sounding, no background noise.",
    "anxious": "{spk} speaks in a calm, reassuring, warm tone, as if steadying someone {she} loves. Very clear audio, close-sounding, no background noise.",
    "lonely": "{spk} speaks in a warm, affectionate, gentle tone, as if keeping someone {she} loves company. Very clear audio, close-sounding, no background noise.",
    "scared": "{spk} speaks in a calm, protective, warm tone, as if reassuring someone {she} loves. Very clear audio, close-sounding, no background noise.",
    "angry": "{spk} speaks in a calm, patient, warm tone, as if gently understanding someone {she} loves. Very clear audio, close-sounding, no background noise.",
    "neutral": "{spk} speaks in a warm, gentle, natural tone, as if chatting with someone {she} loves. Very clear audio, close-sounding, no background noise.",
}


# Telugu captions copied EXACTLY from the teammate's working telugu_pipeline.py
# (an unnamed warm female speaker, chosen by listening; this worked better than a named speaker).
TELUGU_CAPTIONS = {
    "happy": "A warm, affectionate female speaker delivers this in a cheerful, gentle tone, speaking slowly and gently, as if sharing good news with someone she loves. Very clear audio.",
    "sad": "A warm, affectionate female speaker delivers this in a soft, comforting tone, speaking slowly and gently, as if consoling someone she loves. Very clear audio.",
    "anxious": "A warm, affectionate female speaker delivers this in a calm, reassuring tone, speaking slowly and gently, as if steadying someone she loves. Very clear audio.",
    "lonely": "A warm, affectionate female speaker delivers this in a gentle, tender tone, speaking slowly and gently, as if keeping someone she loves company. Very clear audio.",
    "scared": "A warm, affectionate female speaker delivers this in a calm, protective tone, speaking slowly and gently, as if reassuring someone she loves. Very clear audio.",
    "angry": "A warm, affectionate female speaker delivers this in a calm, patient tone, speaking slowly and gently, as if gently understanding someone she loves. Very clear audio.",
    "neutral": "A warm, affectionate female speaker delivers this in a soft conversational tone, speaking slowly and gently, as if comforting someone she loves. Very clear audio.",
}

# danger_words: a small backstop list checked against the transcript on top of
# Gemini's own risk judgement. NOTE: have a native speaker review/extend these.
LANGUAGES = {
    "ta": {
        "name": "Tamil",
        "script": "Tamil script",
        "whisper": "ta",
        "speaker": "Kavitha",
        "she": "she",
        "danger_words": ["சாக வேண்டும்", "சாகணும்", "செத்துவிடலாம்", "செத்துடலாம்", "உயிரை மாய்", "தற்கொலை"],
    },
    "hi": {
        "name": "Hindi",
        "script": "Devanagari script",
        "whisper": "hi",
        "speaker": "Divya",
        "she": "she",
        "danger_words": ["मरना चाहता", "मरना चाहती", "मर जाना चाहता", "मर जाना चाहती", "जान दे दूं", "आत्महत्या", "खुद को नुकसान"],
    },
    "te": {
        "name": "Telugu",
        "script": "Telugu script",
        "whisper": "te",
        "speaker": "Lalitha",
        "she": "she",
        "captions": TELUGU_CAPTIONS,   # tested captions; remove this line to use the named-speaker template instead
        "danger_words": ["చనిపోవాలని", "చచ్చిపోవాలని", "చావాలని", "ఆత్మహత్య"],
    },
    "en": {
        "name": "English",
        "script": "Latin script",
        "whisper": "en",
        "speaker": "Mary",
        "she": "she",
        "danger_words": ["want to die", "kill myself", "end my life", "suicide", "hurt myself", "don't want to live", "dont want to live"],
    },
    "bn": {
        "name": "Bengali",
        "script": "Bengali script",
        "whisper": "bn",
        "speaker": "Aditi",
        "she": "she",
        "danger_words": ["মরে যেতে চাই", "মরতে চাই", "আত্মহত্যা", "বেঁচে থাকতে চাই না"],
    },
    "mr": {
        "name": "Marathi",
        "script": "Devanagari script",
        "whisper": "mr",
        "speaker": "Sunita",
        "she": "she",
        "danger_words": ["मला मरायचे आहे", "मरायचं आहे", "आत्महत्या", "जगायचं नाही"],
    },
    "ml": {
        "name": "Malayalam",
        "script": "Malayalam script",
        "whisper": "ml",
        "speaker": "Anjali",
        "she": "she",
        "danger_words": ["മരിക്കണം", "ചാകണം", "ജീവിതം അവസാനിപ്പിക്കണം", "ആത്മഹത്യ"],
    },
}

# Shown to the model for high-risk moments. Verify the number before demos.
HELPLINE_TEXT = "Tele-MANAS 14416, India's free 24x7 mental-health helpline"


def get_language(code):
    if code not in LANGUAGES:
        raise SystemExit(f"Unknown language '{code}'. Available: {', '.join(LANGUAGES)}")
    return LANGUAGES[code]


def build_captions(code):
    cfg = get_language(code)
    if cfg.get("captions"):          # a language can carry its own tested captions
        caps = dict(cfg["captions"])
        caps.setdefault("neutral", next(iter(caps.values())))
        return {e: caps.get(e, caps["neutral"]) for e in CAPTION_TEMPLATES}
    return {
        emotion: template.format(spk=cfg["speaker"], she=cfg["she"])
        for emotion, template in CAPTION_TEMPLATES.items()
    }


def supported_codes():
    return list(LANGUAGES)


def language_list_text():
    """e.g. 'Tamil (ta), Telugu (te), Hindi (hi), Malayalam (ml)'"""
    return ", ".join(f"{c['name']} ({code})" for code, c in LANGUAGES.items())


def all_danger_words():
    """Union over every language: used as a backstop so code-mixed or mis-detected speech is still caught."""
    words = []
    for cfg in LANGUAGES.values():
        words.extend(cfg["danger_words"])
    return words


def pick_language(probs):
    """
    probs: Whisper's language probabilities {whisper_code: probability}.
    Returns (our_code, probability) for the most likely language WE SUPPORT, or None.
    Restricting to supported languages avoids Whisper picking, say, Urdu for Hindi speech.
    """
    best = None
    for code, cfg in LANGUAGES.items():
        p = probs.get(cfg["whisper"])
        if p is not None and (best is None or p > best[1]):
            best = (code, p)
    return best
