import librosa
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import csv
import urllib.request

# Load YAMNet model from TensorFlow Hub
yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

# Load YAMNet class map and find index for "Laughter"
class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
response = urllib.request.urlopen(class_map_url)
lines = [l.decode('utf-8') for l in response.readlines()]
reader = csv.DictReader(lines)

LAUGHTER_CLASS_INDEX = 13

def detect_laughter(audio_path, threshold=0.04, segment_duration=2.0):
    waveform, sr = librosa.load(audio_path, sr=16000)
    duration = librosa.get_duration(y=waveform, sr=sr)
    step = int(segment_duration * sr)

    timestamps = []

    for start in range(0, len(waveform) - step, step):
        end = start + step
        segment = waveform[start:end]

        if len(segment) < step:
            continue

        segment_tensor = tf.convert_to_tensor(segment, dtype=tf.float32)

        # Run through YAMNet
        scores, embeddings, spectrogram = yamnet_model(segment_tensor)
        scores_np = scores.numpy().squeeze()

        laughter_score = scores_np[:, LAUGHTER_CLASS_INDEX].mean()
        print(f"[{start/sr:.2f}s - {end/sr:.2f}s] Laughter score: {laughter_score:.3f}")

        if laughter_score > threshold:
            start_time = round(start / sr, 2)
            end_time = round(end / sr, 2)
            timestamps.append((start_time, end_time))

    return timestamps
