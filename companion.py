"""
companion.py - ONE multilingual emotional companion for Tamil, Telugu, Hindi, Malayalam
(and any language you add in languages.py).

    child's voice -> detect language -> Gemini (hears the AUDIO: words + tone, remembers past
    chats, checks safety) -> Indic Parler-TTS (reply in the SAME language, matched emotion voice)
    + caregiver alert if distress or several low days in a row.

Put this file NEXT TO tamil_pipeline.py (same folder, same venv, same packages).
It does not change any existing file.

RUN (examples):
    python companion.py --audio child_input.wav                  # language detected automatically
    python companion.py --audio telugu_child_input.wav --no-tts  # fast test without the voice
    python companion.py --lang ta --audio child_input.wav        # force a language
    python companion.py --stt sarvam --audio clip.wav            # Sarvam Saaras listens (needs SARVAM_API_KEY)
    python companion.py --stt whisper --audio clip.wav           # Whisper detects the language + transcribes
    python companion.py --compare-stt --audio clip.wav           # show what each listener heard, side by side
    python companion.py --chat                                   # keep chatting: type a recording path each turn
    python companion.py --forget                                 # erase saved memory

How the language is chosen (--lang auto, the default):
    --stt gemini (default): Gemini listens to the audio and tells us the language.
    --stt sarvam          : Sarvam's Saaras model detects the language and transcribes (Indian-language specialist).
    --stt whisper         : Whisper's language detector picks the most likely SUPPORTED language.

Needs GEMINI_API_KEY set in your terminal (and SARVAM_API_KEY only if you use --stt sarvam). Optional alerts: TELEGRAM_BOT_TOKEN +
TELEGRAM_CHAT_ID, and/or SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_TO (else alerts_log.json only).
"""

import argparse
import os
import sys
from datetime import datetime

import companion_support as sup
from languages import (HELPLINE_TEXT, LANGUAGES, all_danger_words, build_captions,
                       language_list_text, pick_language)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
TTS_MODEL_ID = "ai4bharat/indic-parler-tts"
HELPLINE = os.environ.get("HELPLINE_TEXT", HELPLINE_TEXT)

MIME = {".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4",
        ".ogg": "audio/ogg", ".flac": "audio/flac"}

_cache = {}   # models are loaded once and reused (important for --chat)


# ------------------------------------------------------------------
# Gemini prompt
# ------------------------------------------------------------------
def build_prompt(cfg, context, mood, mode, transcript_text=None):
    """cfg = language config if the language is already known, else None (Gemini will detect it)."""
    if cfg:
        language_rule = f"Language: {cfg['name']}. Write the reply ONLY in {cfg['name']} ({cfg['script']})."
        lang_key = f'"language": always "{[c for c, v in LANGUAGES.items() if v is cfg][0]}".'
        script_for_transcript = cfg["script"]
        reply_lang = cfg["name"]
    else:
        language_rule = (f"The person may speak any of these languages: {language_list_text()}. "
                         "Work out which one they speak and reply ONLY in that same language, in its own script. "
                         "If it is none of these, set language to \"other\" and leave reply empty.")
        lang_key = '"language": the two-letter code of the language they spoke (one of: ' + \
                   ", ".join(LANGUAGES) + ', or "other").'
        script_for_transcript = "its own script"
        reply_lang = "the language they spoke"

    if mode == "gemini":
        step_t = f'"transcript": transcribe the speech in the audio exactly, in {script_for_transcript}.'
        step_v = ('"emotion_from_voice": how their VOICE sounds (trembling, crying, flat, fast, '
                  'cheerful, tired...), as one emotion word.')
        source = "The audio of what they said is attached."
    else:
        step_t = f'"transcript": copy this text exactly: {transcript_text}'
        step_v = '"emotion_from_voice": write "unknown" (no audio is available).'
        source = f"They said (already transcribed): {transcript_text}"

    return f"""
You are a warm, caring companion, like a loving parent or grandparent, talking with a child or an elderly person.
{language_rule}

{source}

What you remember from earlier chats (use naturally and sparingly, for example gently ask how they feel today if they were sad before; never recite old chats; never invent memories):
{context}

{mood}

Return ONE JSON object with exactly these keys:
1. {lang_key}
2. {step_t}
3. "emotion_from_words": one of happy, sad, anxious, lonely, scared, angry, neutral (judged from the words).
4. {step_v}
5. "emotion": the final emotion, one of happy, sad, anxious, lonely, scared, angry, neutral. If the voice and words disagree and the voice sounds distressed, trust the voice.
6. "risk": "none", "concern" or "high".
   high = talk of wanting to die or hurt themselves, being hurt or threatened by someone, being in danger, or a medical emergency.
   concern = strong hopelessness, heavy loneliness, or fear that persists.
   none = everything else.
7. "reply": your spoken reply.

Rules for the reply:
- 2-4 SHORT sentences, in {reply_lang} only. Use commas for natural pauses.
- Write numbers as words, and write the whole reply in one language and one script (no mixing of scripts), so the voice pronounces everything correctly.
- Warm, supportive, matched to the emotion. Respond to what they actually said.
- Do not mention emotion detection, AI, or machines. Do not diagnose them.
- If risk is "high": stay calm and kind, tell them they are not alone and you are here, gently encourage them to tell a trusted family member or adult nearby right now, and you may mention {HELPLINE}. Never give any method or instruction that could cause harm and never argue with them.

Respond with the JSON only.
""".strip()


def ask_gemini(client, cfg, audio_path, context, mood, mode, transcript_text=None):
    prompt = build_prompt(cfg, context, mood, mode, transcript_text)
    if mode == "gemini":
        from google.genai import types
        ext = os.path.splitext(audio_path)[1].lower()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        contents = [types.Part.from_bytes(data=audio_bytes, mime_type=MIME.get(ext, "audio/wav")), prompt]
    else:
        contents = prompt
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config={"response_mime_type": "application/json", "temperature": 0.5},
    )
    return sup.parse_model_json(response.text)


# ------------------------------------------------------------------
# Whisper (only used with --stt whisper)
# ------------------------------------------------------------------
def _whisper_model(size):
    if ("whisper", size) not in _cache:
        import whisper
        print(f"Loading Whisper ({size})...")
        _cache[("whisper", size)] = whisper.load_model(size)
    return _cache[("whisper", size)]


def whisper_detect_language(audio_path, size):
    """Most likely SUPPORTED language according to Whisper -> (code, probability)."""
    import whisper
    model = _whisper_model(size)
    audio = whisper.pad_or_trim(whisper.load_audio(audio_path))
    try:
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    except TypeError:  # older whisper versions
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    return pick_language(probs)


def whisper_transcribe(audio_path, cfg, size):
    import torch
    model = _whisper_model(size)
    result = model.transcribe(audio_path, language=cfg["whisper"], fp16=torch.cuda.is_available())
    return result["text"].strip()


# ------------------------------------------------------------------
# Sarvam Saaras (only used with --stt sarvam / --compare-stt)
# ------------------------------------------------------------------
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saaras:v4")


def to_sarvam_code(code):
    """our 'ta' -> Sarvam 'ta-IN'; no language -> 'unknown' (Sarvam auto-detects)."""
    return f"{code}-IN" if code else "unknown"


def from_sarvam_code(sarvam_code):
    """Sarvam 'hi-IN' -> our 'hi' (None if missing)."""
    return (sarvam_code or "").split("-")[0].lower() or None


def sarvam_transcribe(audio_path, code=None):
    """Returns (transcript, our_language_code_or_None, language_probability_or_None)."""
    import requests
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        sys.exit("SARVAM_API_KEY is not set. Set it in your terminal first (same way as GEMINI_API_KEY).")
    with open(audio_path, "rb") as f:
        r = requests.post(
            SARVAM_STT_URL,
            headers={"api-subscription-key": key},
            files={"file": (os.path.basename(audio_path), f)},
            data={"model": SARVAM_STT_MODEL, "language_code": to_sarvam_code(code)},
            timeout=60,
        )
    if r.status_code != 200:
        sys.exit(f"Sarvam error {r.status_code}: {r.text[:300]}")
    body = r.json()
    return (body.get("transcript") or "").strip(), from_sarvam_code(body.get("language_code")), body.get("language_probability")


def gemini_transcribe_only(client, audio_path):
    from google.genai import types
    ext = os.path.splitext(audio_path)[1].lower()
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=audio_bytes, mime_type=MIME.get(ext, "audio/wav")),
                  "Transcribe this speech exactly as spoken, in its original script. Return only the transcript."],
    )
    return (response.text or "").strip()


def compare_stt(client, audio_path, args):
    """Run every listener on the same recording so you can compare them yourself."""
    print("\n=========== WHAT EACH LISTENER HEARD ===========")
    forced = args.lang if args.lang != "auto" else None
    try:
        print("Gemini :", gemini_transcribe_only(client, audio_path))
    except Exception as exc:
        print("Gemini : failed:", exc)
    if os.environ.get("SARVAM_API_KEY"):
        text, det, prob = sarvam_transcribe(audio_path, forced)
        print(f"Sarvam : {text}   [language: {det}{'' if prob is None else f', {prob:.0%}'}]")
    else:
        print("Sarvam : skipped (SARVAM_API_KEY not set)")
    try:
        code = forced
        if not code:
            det = whisper_detect_language(audio_path, args.whisper_size)
            code = det[0] if det else None
        if code:
            print(f"Whisper: {whisper_transcribe(audio_path, LANGUAGES[code], args.whisper_size)}   [language: {code}, model: {args.whisper_size}]")
        else:
            print("Whisper: could not detect a supported language")
    except Exception as exc:
        print("Whisper: failed:", exc)
    print("(Listen to the recording and judge which one wrote what you actually said.)")


# ------------------------------------------------------------------
# Indic Parler-TTS (same generation code as your tamil_pipeline.py)
# ------------------------------------------------------------------
def _tts():
    if "tts" not in _cache:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Loading Indic Parler-TTS...")
        model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_ID).to(device)
        tok = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
        desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
        _cache["tts"] = (model, tok, desc_tok, device)
    return _cache["tts"]


def speak(text, caption, out_path):
    import soundfile as sf
    import torch
    model, tok, desc_tok, device = _tts()
    description_ids = desc_tok(caption, return_tensors="pt").to(device)
    prompt_ids = tok(text, return_tensors="pt").to(device)
    with torch.inference_mode():
        generation = model.generate(
            input_ids=description_ids.input_ids,
            attention_mask=description_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
        )
    sf.write(out_path, generation.cpu().numpy().squeeze(), model.config.sampling_rate)


# ------------------------------------------------------------------
# One conversation turn
# ------------------------------------------------------------------
def resolve_language(forced, detected_code):
    """Returns the language code to use, or None if unsupported."""
    code = forced if forced and forced != "auto" else detected_code
    return code if code in LANGUAGES else None


def run_turn(client, args, audio_path):
    forced = args.lang if args.lang != "auto" else None
    records = sup.load_memory()
    context = sup.recent_context(records)
    mood = sup.mood_text(sup.mood_summary(records))

    transcript_text, code, lang_note = None, forced, ""
    if args.stt == "whisper":
        if not code:
            det = whisper_detect_language(audio_path, args.whisper_size)
            if det is None:
                print("Could not detect a supported language.")
                return
            code, prob = det
            lang_note = f"(Whisper detection, {prob:.0%} sure)"
            if prob < 0.5:
                lang_note += " LOW CONFIDENCE: use --lang to force the language if this is wrong"
        transcript_text = whisper_transcribe(audio_path, LANGUAGES[code], args.whisper_size)
        print("You said:", transcript_text)
    elif args.stt == "sarvam":
        transcript_text, det_code, prob = sarvam_transcribe(audio_path, code)
        if not code:
            code = resolve_language(None, det_code)
            if not code:
                print(f"\nThat language is not supported yet (Sarvam heard: {det_code}). Supported: {language_list_text()}")
                return
            lang_note = "(Sarvam detection" + ("" if prob is None else f", {prob:.0%} sure") + ")"
        if not transcript_text:
            print("Sarvam returned an empty transcript. Try a clearer recording.")
            return
        print("You said:", transcript_text)

    print("Understanding and replying...")
    cfg = LANGUAGES[code] if code else None
    result = ask_gemini(client, cfg, audio_path, context, mood, "gemini" if args.stt == "gemini" else "text", transcript_text)

    if not code:
        code = resolve_language(None, str(result.get("language", "")).lower().strip())
        lang_note = "(Gemini detection)"
        if not code:
            print(f"\nThat language is not supported yet. Supported: {language_list_text()}")
            return
    cfg = LANGUAGES[code]

    transcript = (transcript_text or str(result.get("transcript", ""))).strip()   # trust the listener's own text
    emotion = sup.normalise_emotion(result.get("emotion"))
    emotion_words = sup.normalise_emotion(result.get("emotion_from_words"))
    emotion_voice = str(result.get("emotion_from_voice", "unknown")).lower().strip()
    reply = str(result.get("reply", "")).strip()
    if not reply:
        print("Gemini did not return a reply.")
        return

    # safety: Gemini's judgement + backstop words from ALL languages
    risk = sup.final_risk(result.get("risk"), transcript, all_danger_words())

    print("\n================ RESULT ================")
    print(f"Language          : {cfg['name']} {lang_note}")
    print("You said          :", transcript)
    print("Emotion (words)   :", emotion_words)
    print("Emotion (voice)   :", emotion_voice)
    print("Emotion (final)   :", emotion)
    print("Safety risk       :", risk)
    print("Reply             :", reply)

    now = datetime.now()
    records = sup.add_record({
        "timestamp": now.isoformat(timespec="seconds"),
        "language": cfg["name"],
        "language_code": code,
        "transcript": transcript,
        "emotion_from_words": emotion_words,
        "emotion_from_voice": emotion_voice,
        "emotion": emotion,
        "risk": risk,
        "reply": reply,
    })
    summary = sup.mood_summary(records, now)
    print(f"Mood trend        : {summary['trend']} | low-mood streak: {summary['streak_low_days']} day(s)")

    reasons = sup.alerts_to_send(risk, summary, sup.load_alert_log(), now)
    if reasons:
        entry = sup.send_alert(reasons, summary, cfg["name"], snippet=transcript, now=now)
        print("ALERT raised      :", reasons, "->", entry["delivered_via"] or "logged only (alerts_log.json)")

    if args.no_tts:
        print("\n(--no-tts: voice skipped)")
        return
    captions = build_captions(code)
    out_path = f"reply_{code}.wav"
    speak(reply, captions.get(emotion, captions["neutral"]), out_path)
    print("\nVoice saved as:", out_path)
    try:
        os.startfile(out_path)  # Windows auto-play
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Multilingual emotional companion")
    parser.add_argument("--lang", default="auto", help="auto (default) or a code: " + ", ".join(LANGUAGES))
    parser.add_argument("--audio", default=None, help="input recording")
    parser.add_argument("--stt", choices=["gemini", "sarvam", "whisper"], default="gemini",
                        help="gemini = hears audio incl. tone (default); sarvam = Indian-language specialist; whisper = text only")
    parser.add_argument("--whisper-size", default="small", help="small, medium, large-v3 ...")
    parser.add_argument("--compare-stt", action="store_true", help="show what Gemini, Sarvam and Whisper each heard (no reply, no memory)")
    parser.add_argument("--chat", action="store_true", help="keep chatting; type a recording path each turn")
    parser.add_argument("--no-tts", action="store_true", help="skip voice generation (fast test)")
    parser.add_argument("--forget", action="store_true", help="erase saved memory and exit")
    args = parser.parse_args()

    if args.forget:
        sup.clear_memory()
        print("Memory erased.")
        return
    if args.lang != "auto" and args.lang not in LANGUAGES:
        sys.exit(f"Unknown --lang '{args.lang}'. Use auto or: {', '.join(LANGUAGES)}")
    if not args.chat and not args.audio:
        sys.exit("Give --audio <file> (or use --chat).")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not set. Set it in your terminal first.")
    from google import genai
    client = genai.Client(api_key=api_key)

    if not args.chat:
        if not os.path.isfile(args.audio):
            sys.exit(f"Audio file not found: {args.audio}")
        if args.compare_stt:
            compare_stt(client, args.audio, args)
            return
        run_turn(client, args, args.audio)
        return

    print("Chat mode. Type the path of a recording each turn (press Enter on an empty line to quit).")
    while True:
        path = input("\nRecording file > ").strip().strip('"')
        if not path:
            break
        if not os.path.isfile(path):
            print("File not found.")
            continue
        run_turn(client, args, path)


if __name__ == "__main__":
    main()
