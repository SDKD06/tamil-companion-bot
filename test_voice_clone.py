"""
Simple test: IndicF5 voice cloning with your reference voice.

SETUP (run once, in your activated tts-env):
    pip install git+https://github.com/ai4bharat/IndicF5.git
    pip install soundfile torch torchaudio transformers

RUN:
    python test_voice_clone.py
"""

from transformers import AutoModel
import soundfile as sf

REFERENCE_VOICE = "reference_voice.wav"   # <-- your reference file, same folder as this script
REFERENCE_TEXT = "கண்ணா,சாப்பிட்டியா?இன்னிக்கு உன் நாள் எப்படி போச்சு?அம்மா இங்கதான் இருக்கேன்.எல்லாம் சரி ஆயிடும்."  # <-- the text that was spoken in your reference voice
# The Tamil sentence you want it to speak, in the cloned voice
text_to_speak = "வணக்கம்"

print("Loading IndicF5 model (first run will download it)...")
model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)

print("Generating cloned speech...")
audio = model(
    text_to_speak,
    ref_audio_path=REFERENCE_VOICE,
    ref_text=REFERENCE_TEXT,
)

sf.write("cloned_output.wav", audio, samplerate=24000)
print("\nDone! Saved as: cloned_output.wav")
