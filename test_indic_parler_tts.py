"""
Test script: Indic Parler-TTS for Tamil (warm/motherly voice)

SETUP (run these once, in Colab or your terminal):
    pip install git+https://github.com/huggingface/parler-tts.git
    pip install soundfile

RUN:
    python test_indic_parler_tts.py

This generates 3 .wav files so you can compare different style captions
side by side and pick the one that sounds closest to what you want.
"""

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load model + tokenizers once
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)


def generate(prompt_text: str, description_text: str, out_path: str):
    """Generate one audio file from Tamil text + a style description."""
    description_input_ids = description_tokenizer(description_text, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)

    generation = model.generate(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask,
    )
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(out_path, audio_arr, model.config.sampling_rate)
    print(f"Saved: {out_path}")


# ---- A sample Tamil sentence to test with (edit this to something Gemini would actually say) ----
tamil_text = "நீங்க எப்படி இருக்கீங்க? நான் இங்க இருக்கேன், கவலைப்படாதீங்க."
# (roughly: "How are you? I'm here, don't worry.")

# ---- Try a few different warmth-focused captions and compare ----

# 1. Using the recommended Tamil speaker "Jaya", warm and calm
generate(
    tamil_text,
    "Jaya speaks with a warm, gentle, and caring tone at a slow, calm pace. "
    "The recording is very clear audio, close-sounding, with no background noise.",
    "test1_jaya_warm.wav",
)

# 2. Explicit "Conversation" emotion style (Tamil supports this officially)
generate(
    tamil_text,
    "A warm, affectionate female speaker delivers this in a soft conversational tone, "
    "speaking slowly and gently, as if comforting someone she loves. Very clear audio.",
    "test2_conversation_style.wav",
)

# 3. "Happy" emotion style, for comparison (in case a lighter tone fits better sometimes)
generate(
    tamil_text,
    "A cheerful, warm female speaker speaks with gentle happiness and encouragement, "
    "at a moderate pace, in a very clear, close-sounding recording.",
    "test3_happy_style.wav",
)


generate(
    tamil_text,
    "Jaya speaks in a very soft, hushed, tender voice, almost like a gentle whisper, "
    "at a slow and unhurried pace, soothing and low in volume. "
    "Very clear audio, close-sounding, with no background noise.",
    "test4_soft.wav",
)

generate(
    tamil_text,
    "A gentle, breathy female voice speaks softly and tenderly, as if quietly comforting "
    "someone right beside her, speaking slowly with a soothing, low volume. Very clear audio.",
    "test5_soft_v2.wav",
)

print("\nDone. Listen to all three and note which caption style sounds most 'motherly' to you.")
print("Once you pick one, hardcode that exact description string as a constant in your pipeline.")