import sounddevice as sd
import numpy as np
import torch
import librosa

from panns_inference.inference import AudioTagging


SAMPLE_RATE = 32000
CHUNK_DURATION = 2.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)


print("Loading PANNs AudioTagging model...")
model = AudioTagging(
    checkpoint_path=None,     
    device='cpu'             
)
print("Model loaded successfully.")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    audio = indata[:, 0]


    if np.max(np.abs(audio)) < 0.01:
        return

    if len(audio) != CHUNK_SAMPLES:
        audio = librosa.resample(audio, orig_sr=SAMPLE_RATE, target_sr=32000)

    audio = audio[np.newaxis, :]

    with torch.no_grad():
        clipwise_output, labels = model.inference(audio)

   
    top_indices = np.argsort(clipwise_output[0])[::-1][:5]

    print("\n🔊 Detected sounds:")
    for idx in top_indices:
        score = clipwise_output[0][idx]
        if score > 0.1:
            print(f"  {labels[idx]} ({score:.2f})")
    print("-" * 40)


print("🎧 Live Audio Detection Started")
print("Make sounds: dog bark, speech, clap, engine, etc.")
print("Press Ctrl+C to stop\n")

with sd.InputStream(
    channels=1,
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK_SAMPLES,
    callback=audio_callback
):
    while True:
        sd.sleep(1000)
