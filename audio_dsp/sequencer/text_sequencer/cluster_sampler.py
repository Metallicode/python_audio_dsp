import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import umap
from sklearn.preprocessing import StandardScaler
import pygame
import io
import tkinter as tk
import threading

# Parameters
SAMPLE_DIR = "cluster_samples"  # Folder with .wav files
SAMPLE_RATE = 44100
SPECTROGRAM_SIZE = (100, 100)  # Resize spectrograms for consistency

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
        if file.endswith('.wav'):
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
    print(f"UMAP embedding shape: {embedding.shape}, min: {embedding.min()}, max: {embedding.max()}")
    return embedding

def create_composite_image(embedding, images, samples):
    print("Step 1: Importing Tkinter...")
    import tkinter as tk  # Double-check import
    print("Step 2: Initializing Tkinter root...")
    root = tk.Tk()
    print("Step 3: Setting title...")
    root.title("Spectrogram Map")

    print("Step 4: Defining canvas dimensions...")
    canvas_width = 800
    canvas_height = 600
    print("Step 5: Creating canvas...")
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
    print("Step 6: Packing canvas...")
    canvas.pack(pady=10)

    print("Step 7: Normalizing coordinates...")
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    print(f"X range: {x_min} to {x_max}, Y range: {y_min} to {y_max}")
    scale_x = (canvas_width - 2 * SPECTROGRAM_SIZE[0]) / (x_max - x_min)
    scale_y = (canvas_height - 2 * SPECTROGRAM_SIZE[1]) / (y_max - y_min)

    print("Step 8: Preparing to process images...")
    photo_images = []
    sample_positions = []
    for i, (x, y) in enumerate(embedding):
        canvas_x = (x - x_min) * scale_x + SPECTROGRAM_SIZE[0] / 2
        canvas_y = (y - y_min) * scale_y + SPECTROGRAM_SIZE[1] / 2
        img = images[i]
        print(f"Step 9.{i}: Processing sample {samples[i]}, position ({canvas_x:.2f}, {canvas_y:.2f})")
        try:
            photo = ImageTk.PhotoImage(image=img)
            print(f"Step 9.{i}: Created PhotoImage for sample {i}")
        except Exception as e:
            print(f"Step 9.{i}: Failed to create PhotoImage: {e}")
            continue
        canvas.create_image(canvas_x, canvas_y, image=photo)
        photo_images.append(photo)
        sample_positions.append((canvas_x, canvas_y, samples[i]))

    def play_sample(event):
        x, y = event.x, event.y
        for pos_x, pos_y, sample_path in sample_positions:
            if abs(pos_x - x) < SPECTROGRAM_SIZE[0] / 2 and abs(pos_y - y) < SPECTROGRAM_SIZE[1] / 2:
                print(f"Playing: {sample_path}")
                threading.Thread(target=lambda: pygame.mixer.Sound(sample_path).play()).start()
                break

    print("Step 10: Binding motion event...")
    canvas.bind("<Motion>", play_sample)
    
    print("Step 11: Starting Tkinter mainloop...")
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

    print("Creating interactive map...")
    create_composite_image(embedding, images, samples)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")