import whisper
import os
import time

# 📁 Dossier contenant les fichiers .mp3
audio_folder = "./samples"

# 📦 Charger le modèle Whisper (tu peux changer "base" par "tiny", "small", etc.)
model = whisper.load_model("large-v3")

def is_segment_to_filter(segment):
    """Check if segment contains usefull transcription"""
    return segment["no_speech_prob"] >= 0.8 or segment["temperature"] > 0.2

for filename in os.listdir(audio_folder):
    if filename.lower().endswith(".mp3"):
        file_path = os.path.join(audio_folder, filename)
        print(f"Transcription de : {filename}...")

        # 🧠 Transcription
        start = time.time()
        result = model.transcribe(file_path)
        segments = result["segments"]
        language = result["language"]
        transcription = ""
        for seg in segments:
            if is_segment_to_filter(seg):
                continue
            transcription += " " + seg["text"]
            
        if transcription == "":
            continue
        print(f"Detected language: {language}")
        print(f"Transcript: {transcription}")
        print(f"✅ Transcrit en {str(time.time() - start)} s")

