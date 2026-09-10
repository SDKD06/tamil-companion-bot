"""
Tamil Companion Bot Pipeline (Final)
    Whisper (speech->text) -> Gemini (detects emotion + warm supportive reply)
    -> Indic Parler-TTS (speaks the reply, with tone matched to detected emotion)

SETUP (run once, in your activated tts-env):
    pip install openai-whisper
    pip install git+https://github.com/huggingface/parler-tts.git
    pip install google-genai soundfile torch torchaudio

RUN:
    python tamil_pipeline.py
"""

import os
import json
from datetime import datetime

import whisper
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from google import genai

# ============================================================
# CONFIGURATION
# ============================================================
import os
API_KEY = os.environ.get("GEMINI_API_KEY")
AUDIO_FILE = "child_input.wav"   # Tamil speech from the child/elder to test with
HISTORY_FILE = "conversation_history.json"

# Map detected emotion -> Indic Parler-TTS style caption
# Edit these descriptions freely as you keep tuning the tone
EMOTION_CAPTIONS = {
    "happy": (
        "Kavitha speaks in a warm, cheerful, gentle tone, "
        "as if sharing good news with someone she loves. Very clear audio, close-sounding, no background noise."
    ),
    "sad": (
        "Kavitha speaks in a soft, comforting, warm tone, "
        "as if consoling someone she loves. Very clear audio, close-sounding, no background noise."
    ),
    "anxious": (
        "Kavitha speaks in a calm, reassuring, warm tone, "
        "as if steadying someone she loves. Very clear audio, close-sounding, no background noise."
    ),
    "lonely": (
        "Kavitha speaks in a warm, affectionate, gentle tone, "
        "as if keeping someone she loves company. Very clear audio, close-sounding, no background noise."
    ),
    "scared": (
        "Kavitha speaks in a calm, protective, warm tone, "
        "as if reassuring someone she loves. Very clear audio, close-sounding, no background noise."
    ),
    "angry": (
        "Kavitha speaks in a calm, patient, warm tone, "
        "as if gently understanding someone she loves. Very clear audio, close-sounding, no background noise."
    ),
    "neutral": (
        "Kavitha speaks in a warm, gentle, natural tone, "
        "as if chatting with someone she loves. Very clear audio, close-sounding, no background noise."
    ),
}
DEFAULT_CAPTION = EMOTION_CAPTIONS["neutral"]

# ============================================================
# CLIENTS / MODELS
# ============================================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"

client = genai.Client(api_key=API_KEY)

print("Loading Whisper...")
whisper_model = whisper.load_model("small")

print("Loading Indic Parler-TTS...")
tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)


# ============================================================
# SAVE CONVERSATION
# ============================================================
def save_conversation(user_text, emotion, ai_reply):
    item = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "recognized_text": user_text,
        "emotion": emotion,
        "ai_reply": ai_reply,
    }
    data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(item)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# ============================================================
# SPEECH TO TEXT (Tamil)
# ============================================================
print("\nTranscribing...")
result = whisper_model.transcribe(AUDIO_FILE, language="ta")
transcript = result["text"].strip()
print("\n==================== TRANSCRIPTION ====================")
print(transcript)

# ============================================================
# EMOTION DETECTION + SUPPORTIVE REPLY (Gemini, combined)
# ============================================================
print("\nGenerating supportive AI response...")
prompt = f"""
You are a caring and supportive parent, replying in Tamil to a child or elderly person.

Their message (in Tamil):
{transcript}

Step 1: Identify their emotional state as exactly one of:
happy, sad, anxious, lonely, scared, angry, neutral

Step 2: Reply naturally and supportively, like a caring parent, in Tamil,
in a way that responds appropriately to that emotion.

Rules for the reply:
- 2-4 sentences, in Tamil only.
- Warm, supportive, and emotionally appropriate to what they said.
- Use commas to create natural pauses where a caring person would pause while speaking.
- Keep sentences short — avoid long unbroken sentences.
- Do not mention the emotion detection.
- Do not say things like "I sense you are sad."
- Do not mention AI or machine learning.

Respond ONLY in this exact JSON format, nothing else:
{{"emotion": "...", "reply": "..."}}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
)

raw = response.text.strip().replace("```json", "").replace("```", "").strip()
parsed = json.loads(raw)
emotion = parsed["emotion"].lower().strip()
ai_reply = parsed["reply"]

print("\n==================== AI REPLY ====================")
print(f"Detected emotion: {emotion}")
print(ai_reply)

save_conversation(transcript, emotion, ai_reply)
print(f"\nConversation saved to: {HISTORY_FILE}")

# ============================================================
# GENERATE VOICE (Indic Parler-TTS, tone matched to emotion)
# ============================================================
print("\nGenerating voice response...")
caption = EMOTION_CAPTIONS.get(emotion, DEFAULT_CAPTION)

description_input_ids = description_tokenizer(caption, return_tensors="pt").to(device)
prompt_input_ids = tts_tokenizer(ai_reply, return_tensors="pt").to(device)

generation = tts_model.generate(
    input_ids=description_input_ids.input_ids,
    attention_mask=description_input_ids.attention_mask,
    prompt_input_ids=prompt_input_ids.input_ids,
    prompt_attention_mask=prompt_input_ids.attention_mask,
)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("reply.wav", audio_arr, tts_model.config.sampling_rate)

print("\nParent voice saved as: reply.wav")

try:
    os.startfile("reply.wav")
except Exception:
    pass

# ============================================================
# FINAL RESULT
# ============================================================
print("\n==========================================================")
print(" FINAL RESULT")
print("==========================================================")
print("\nRecognized Text:")
print(transcript)
print("\nDetected Emotion:")
print(emotion)
print("\nAI Parent Reply:")
print(ai_reply)
print("\nOutput Audio:")
print("reply.wav")
print("\nFinished.")
