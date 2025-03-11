from fastapi import FastAPI
import deepspeech
import numpy as np
import pyaudio

# Initialize FastAPI app
app = FastAPI()

# Load DeepSpeech model
model_path = "deepspeech-0.9.3-models.pbmm"
scorer_path = "deepspeech-0.9.3-models.scorer"

model = deepspeech.Model(model_path)
model.enableExternalScorer(scorer_path)

# Audio settings
RATE = 16000  # DeepSpeech requires 16kHz mono audio
CHUNK = 1024  # Buffer size for real-time audio

# Initialize PyAudio
p = pyaudio.PyAudio()

@app.get("/")
def home():
    return {"message": "Welcome to the DeepSpeech FastAPI server!"}

@app.get("/transcribe")
def transcribe_speech():
    """Captures audio from the microphone and returns transcribed text"""
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("🎤 Listening... Speak now!")
    frames = []

    # Capture 5 seconds of audio
    for _ in range(0, int(RATE / CHUNK * 5)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()

    # Convert recorded audio to numpy array
    audio_data = np.concatenate(frames, axis=0)

    # Transcribe speech
    text = model.stt(audio_data)
    return {"transcription": text}

