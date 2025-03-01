import numpy as np
import librosa
import soundfile as sf
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def load_custom_image(file_path, size=256):
    img = Image.open(file_path)
    img = img.convert('L')
    try:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    except AttributeError:
        img = img.resize((size, size), Image.LANCZOS)
    img_array = np.array(img, dtype=np.uint8)
    return img_array

def apply_image_effect(img, effect="coarse", param=8):
    """
    Apply effects to the image.
    - effect: 'coarse', 'jagged', 'shifted', 'contrast', 'invert'
    - param: Effect-specific parameter
    """
    h, w = img.shape
    
    if effect == "coarse":
        small = cv2.resize(img, (w // param, h // param), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    elif effect == "jagged":
        n_blocks_h = (h + param - 1) // param
        n_blocks_w = (w + param - 1) // param
        block_sizes = np.random.randint(param // 2, param, (n_blocks_h, n_blocks_w))
        jagged = np.zeros_like(img)
        for i in range(0, h, param):
            for j in range(0, w, param):
                block_h = min(param, h - i)
                block_w = min(param, w - j)
                block_size = block_sizes[i // param, j // param]
                small_block = cv2.resize(img[i:i+block_h, j:j+block_w], 
                                        (block_size, block_size), interpolation=cv2.INTER_NEAREST)
                jagged[i:i+block_h, j:j+block_w] = cv2.resize(small_block, 
                                                             (block_w, block_h), interpolation=cv2.INTER_NEAREST)
        return jagged
    
    elif effect == "shifted":
        small = cv2.resize(img, (w // param, h // param), interpolation=cv2.INTER_NEAREST)
        shifted = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        shift_max = param // 2
        for i in range(0, h, param):
            for j in range(0, w, param):
                block_h = min(param, h - i)
                block_w = min(param, w - j)
                shift_x = np.random.randint(-shift_max, shift_max)
                shift_y = np.random.randint(-shift_max, shift_max)
                src_y, src_x = i, j
                dst_y = np.clip(i + shift_y, 0, h - block_h)
                dst_x = np.clip(j + shift_x, 0, w - block_w)
                shifted[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = shifted[src_y:src_y+block_h, src_x:src_x+block_w]
        return shifted
    
    elif effect == "contrast":
        # Adjust contrast (param > 1 increases, < 1 decreases)
        adjusted = np.clip(img * param, 0, 255).astype(np.uint8)
        return adjusted
    
    elif effect == "invert":
        # Invert pixel values (no param needed, but included for consistency)
        return 255 - img
        
    elif effect == "no_effect":
        return img
    
    else:
        raise ValueError("Effect must be 'coarse', 'jagged', 'shifted', 'contrast', or 'invert'")

def show_image(img, title="Generated Image"):
    plt.imshow(img, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

def image_to_rhythmic_audio(img, sr=44100, duration=2.0):
    size = img.shape[0]
    n_fft = min(2048, size // 2)
    hop_length = n_fft // 4
    num_frames = int(duration * sr / hop_length)
    spec = cv2.resize(img, (num_frames, n_fft // 2 + 1), interpolation=cv2.INTER_LINEAR)
    mag = spec / 255.0
    mag = mag ** 2  # Sharpen peaks for rhythm
    mag = librosa.db_to_amplitude(librosa.amplitude_to_db(mag, ref=1.0) * 2)
    phase = np.random.uniform(-np.pi, np.pi, mag.shape)
    spec_complex = mag * np.exp(1j * phase)
    audio = librosa.istft(spec_complex, hop_length=hop_length, length=int(duration * sr))
    audio = librosa.util.normalize(audio)
    return np.clip(audio, -1.0, 1.0), sr

if __name__ == "__main__":
    custom_image_path = "x.jpg"  # Replace with your image path
    
    effects = [
        ("no_effect", 0, "No Effect")
        ("coarse", 8, "Coarse Pixelated"),
        ("jagged", 10, "Jagged Pixelated"),
        ("shifted", 12, "Shifted Pixelated"),
        ("contrast", 1.5, "High Contrast"),  # Boost contrast
        ("contrast", 0.5, "Low Contrast"),   # Reduce contrast
        ("invert", 1, "Inverted")            # Param ignored for invert
    ]
    
    output_files = []
    base_img = load_custom_image(custom_image_path, size=256)
    show_image(base_img, "Custom Image")
    
    for effect, param, title in effects:
        effected_img = apply_image_effect(base_img, effect=effect, param=param)
        show_image(effected_img, f"{title} Custom Image")
        audio, sr = image_to_rhythmic_audio(effected_img, sr=44100, duration=2.0)
        output_file = f"custom_{effect}_{param if effect != 'invert' else 'inv'}.wav"
        sf.write(output_file, audio, sr, subtype='PCM_16')
        output_files.append(output_file)
    
    print("Generated rhythmic audio files from custom image:", ", ".join(output_files))