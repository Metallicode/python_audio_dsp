import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pygame
import io
import tkinter as tk
import threading
import json

# Parameters
SAMPLE_DIR = "cluster_samples"  # Folder with .wav files
SAMPLE_RATE = 44100
SPECTROGRAM_SIZE = (100, 100)  # Resize spectrograms for consistency
NUM_CLUSTERS = 4  # Default number of clusters

# Initialize pygame for audio
pygame.mixer.init()

def audio_to_spectrogram(file_path):
    print(f"Converting {file_path} to spectrogram...")
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig, ax = plt.subplots(figsize=(2, 2))
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).resize(SPECTROGRAM_SIZE).convert('RGB')
    plt.close(fig)
    print(f"Spectrogram generated: {img.size}, mode {img.mode}")
    return np.array(img.convert('L')).flatten(), img

def load_samples(directory):
    print("Loading samples from directory...")
    samples = []
    spectrograms = []
    images = []
    for file in os.listdir(directory):
        if file.endswith('.wav') or file.endswith('.WAV'):
            path = os.path.join(directory, file)
            spec_data, spec_img = audio_to_spectrogram(path)
            samples.append(path)
            spectrograms.append(spec_data)
            images.append(spec_img)
    print(f"Loaded {len(samples)} samples")
    return samples, np.array(spectrograms), images

def map_spectrograms(spectrograms):
    print("Mapping spectrograms with UMAP...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(spectrograms)
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(scaled_data)
    print(f"UMAP embedding shape: {embedding.shape}")
    return embedding

def cluster_spectrograms(spectrograms, k=NUM_CLUSTERS):
    print(f"Clustering spectrograms into {k} clusters...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(spectrograms)
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    print(f"Cluster labels: {cluster_labels}")
    return cluster_labels

def generate_cluster_mapping(samples, cluster_labels):
    """Generate a readable mapping of samples to clusters with string keys."""
    cluster_map = {}
    for sample, label in zip(samples, cluster_labels):
        label_str = str(label)  # Convert int32 to string
        if label_str not in cluster_map:
            cluster_map[label_str] = []
        cluster_map[label_str].append(sample)
    return cluster_map

def create_composite_image(embedding, images, samples, cluster_labels):
    root = tk.Tk()
    root.title("Spectrogram Map with Clusters")

    canvas_width = 800
    canvas_height = 600
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack(pady=10)

    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    scale_x = (canvas_width - 2 * SPECTROGRAM_SIZE[0]) / (x_max - x_min)
    scale_y = (canvas_height - 2 * SPECTROGRAM_SIZE[1]) / (y_max - y_min)

    photo_images = []
    sample_positions = []
    for i, (x, y) in enumerate(embedding):
        canvas_x = (x - x_min) * scale_x + SPECTROGRAM_SIZE[0] / 2
        canvas_y = (y - y_min) * scale_y + SPECTROGRAM_SIZE[1] / 2
        img = images[i]
        photo = ImageTk.PhotoImage(image=img)
        canvas.create_image(canvas_x, canvas_y, image=photo)
        photo_images.append(photo)
        sample_positions.append((canvas_x, canvas_y, samples[i], cluster_labels[i]))

    def play_sample(event):
        x, y = event.x, event.y
        for pos_x, pos_y, sample_path, cluster in sample_positions:
            if abs(pos_x - x) < SPECTROGRAM_SIZE[0] / 2 and abs(pos_y - y) < SPECTROGRAM_SIZE[1] / 2:
                print(f"Playing: {sample_path} (Cluster {cluster})")
                threading.Thread(target=lambda: pygame.mixer.Sound(sample_path).play()).start()
                break

    canvas.bind("<Motion>", play_sample)
    
    root.mainloop()

def main():
    print("Starting main process...")
    print("Loading samples...")
    samples, spectrograms, images = load_samples(SAMPLE_DIR)
    if not samples:
        print(f"No .wav files found in {SAMPLE_DIR}")
        return

    print("Mapping spectrograms...")
    embedding = map_spectrograms(spectrograms)

    print("Clustering samples...")
    cluster_labels = cluster_spectrograms(spectrograms, k=NUM_CLUSTERS)
    cluster_map = generate_cluster_mapping(samples, cluster_labels)

    print("\nCluster Mapping:")
    for cluster_id, sample_list in cluster_map.items():
        print(f"Cluster {cluster_id}:")
        for sample in sample_list:
            print(f"  - {sample}")

    # Save mapping to JSON
    with open("cluster_mapping.json", "w") as f:
        json.dump(cluster_map, f, indent=4)
    print("Cluster mapping saved to 'cluster_mapping.json'")

    print("Creating interactive map...")
    create_composite_image(embedding, images, samples, cluster_labels)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")