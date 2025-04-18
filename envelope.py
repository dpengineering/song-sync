import librosa
import numpy as np
import json

# 1️⃣ Load your audio
y, sr = librosa.load('/home/softdev/Documents/song-sync/five_hours.mp4', sr=None)

# 2️⃣ Choose your hop size (e.g., 512 samples → ~11.6 ms at 44.1 kHz)
hop_length = 4096

# 3️⃣ Compute the amplitude envelope (peak per frame)
envelope = []
for i in range(0, len(y), hop_length):
    frame = y[i:i+hop_length]
    envelope.append(np.max(np.abs(frame)))

envelope = np.array(envelope)

# 4️⃣ Optionally smooth it (moving average)
window = 5  # frames
kernel = np.ones(window) / window
envelope_smooth = np.convolve(envelope, kernel, mode='same')

# 5️⃣ Normalize to 0…1
env_min, env_max = envelope_smooth.min(), envelope_smooth.max()
envelope_norm = (envelope_smooth - env_min) / (env_max - env_min)

# 6️⃣ Export to JSON (or directly to a JS file)
data = {
    'hop_time': hop_length / sr,       # seconds per envelope sample
    'envelope': envelope_norm.tolist()
}

with open('envelope.json', 'w') as f:
    json.dump(data, f)
print(f"Saved envelope ({len(envelope_norm)} frames) → envelope.json")
