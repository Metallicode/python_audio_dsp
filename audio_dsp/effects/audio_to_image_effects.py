import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, rotate
import cv2
from audio_dsp.utils import load_audio, normalize_audio

def audio_to_image(file, size=256):
    sr, audio = load_audio(file, mono=True)
    n_fft = 2048
    hop_length = 512
    spec = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(spec)
    mag = librosa.amplitude_to_db(mag, ref=np.max)
    mag = (mag - mag.min()) / (mag.max() - mag.min()) * 255
    mag = mag.astype(np.uint8)
    mag_resized = cv2.resize(mag, (size, size), interpolation=cv2.INTER_LINEAR)
    return mag_resized, audio, sr, n_fft, hop_length

def show_image(img, title="Audio Image"):
    plt.imshow(img, cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

def apply_visual_effects(img, effect="blur", param=5):
    if effect == "blur":
        # Smooths frequencies/time
        return gaussian_filter(img, sigma=param)
    elif effect == "swirl":
        # Warps polar-style
        rows, cols = img.shape
        center = (cols // 2, rows // 2)
        swirled = cv2.warpPolar(img, (rows, cols), center, max(rows, cols), 
                               cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)
        return swirled
    elif effect == "rotate":
        # Shifts time/frequency
        return rotate(img, angle=param, reshape=False)
    elif effect == "contrast":
        # Amplifies peaks
        return np.clip(img * param, 0, 255).astype(np.uint8)
    elif effect == "pixelate":
        # Blocky, glitchy effect
        h, w = img.shape
        small = cv2.resize(img, (w // param, h // param), interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    elif effect == "wave":
        # Wavy distortion
        rows, cols = img.shape
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(param * np.sin(2 * np.pi * i / 128))
                offset_y = int(param * np.cos(2 * np.pi * j / 128))
                if i + offset_y < rows and j + offset_x < cols:
                    img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
        return img_output
    elif effect == "edge":
        # Highlights sharp changes (like a high-pass filter)
        edges = cv2.Canny(img, 100, 200)
        return np.clip(img + edges * param, 0, 255).astype(np.uint8)
    elif effect == "invert":
        # Flips amplitude polarity visually
        return 255 - img
        
    elif effect == "ripple":
        # Circular ripple effect radiating from center
        rows, cols = img.shape
        center = (cols // 2, rows // 2)
        x, y = np.indices((rows, cols))
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        ripple = img * (1 + param * np.sin(dist / 10))
        return np.clip(ripple, 0, 255).astype(np.uint8)
        # Audio: Pulsing, ring-mod-like oscillations across frequencies and time

    elif effect == "shear":
        # Shears the image horizontally or vertically
        rows, cols = img.shape
        shear_matrix = np.float32([[1, param, 0], [0, 1, 0]])  # Horizontal shear
        sheared = cv2.warpAffine(img, shear_matrix, (cols, rows))
        return sheared
        # Audio: Slanted time-frequency shifts, like a stretched or skewed delay

    elif effect == "threshold":
        # Hard cutoff at a brightness level
        _, thresh = cv2.threshold(img, param * 255, 255, cv2.THRESH_BINARY)
        return thresh
        # Audio: Harsh, clipped texture—turns subtle gradients into stark on/off switches

    elif effect == "emboss":
        # Emboss filter to highlight edges in relief
        kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]) * param
        embossed = cv2.filter2D(img, -1, kernel) + 128  # Offset to avoid negatives
        return np.clip(embossed, 0, 255).astype(np.uint8)
        # Audio: Exaggerated transients and contours, like a metallic, sculpted sound

    elif effect == "solarize":
        # Inverts values above a threshold
        solarized = np.where(img > param * 255, 255 - img, img)
        return solarized.astype(np.uint8)
        # Audio: Partial inversion—creates a dynamic, unpredictable amplitude flip

    elif effect == "posterize":
        # Reduces color depth for a banded look
        levels = int(param) if param >= 2 else 2  # At least 2 levels
        posterized = (img // (256 // levels)) * (256 // levels)
        return posterized.astype(np.uint8)
        # Audio: Stepped, quantized feel—like a low-bit-depth digital crunch
        
    elif effect == "jagged_pixelate":
        # Pixelate with random block sizes
        h, w = img.shape
        block_sizes = np.random.randint(2, param, (h // param, w // param))
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
        # Audio: Uneven, gritty glitches—like a broken digital signal with varying resolution.

    elif effect == "shifted_pixelate":
        # Pixelate with random block shifts
        h, w = img.shape
        small = cv2.resize(img, (w // param, h // param), interpolation=cv2.INTER_NEAREST)
        shifted = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        shift_max = param // 2
        for i in range(0, h, param):
            for j in range(0, w, param):
                shift_x = np.random.randint(-shift_max, shift_max)
                shift_y = np.random.randint(-shift_max, shift_max)
                src_y, src_x = i, j
                dst_y = np.clip(i + shift_y, 0, h - param)
                dst_x = np.clip(j + shift_x, 0, w - param)
                shifted[dst_y:dst_y+param, dst_x:dst_x+param] = shifted[src_y:src_y+param, src_x:src_x+param]
        return shifted
        # Audio: Stuttering, displaced glitches—like audio chunks sliding out of sync.     
    elif effect == "mirror":
        # Mirror effect: flip horizontally, vertically, or both
        if param == "horizontal":
            mirrored = cv2.flip(img, 1)  # Flip left-right
        elif param == "vertical":
            mirrored = cv2.flip(img, 0)  # Flip top-bottom
        elif param == "both":
            mirrored = cv2.flip(img, -1)  # Flip both axes
        else:
            raise ValueError("Mirror param must be 'horizontal', 'vertical', or 'both'")
        return mirrored
        # Audio:
        # - Horizontal: Reverses time—like a backwards playback.
        # - Vertical: Inverts frequencies—highs to lows, lows to highs.
        # - Both: Backwards and frequency-inverted—total audio mind-bend.
        
        
    elif effect == "column_shuffle":
        # Split into n columns and rearrange
        if not isinstance(param, tuple) or len(param) != 2:
            raise ValueError("Column_shuffle param must be a tuple: (n_columns, mode)")
        n_columns, mode = param
        if not isinstance(n_columns, int) or n_columns < 1:
            raise ValueError("n_columns must be a positive integer")
        if mode not in ["random", "intensity"]:
            raise ValueError("Mode must be 'random' or 'intensity'")
        
        h, w = img.shape
        column_width = w // n_columns
        columns = []
        
        # Split into columns
        for i in range(0, w, column_width):
            col_end = min(i + column_width, w)
            columns.append(img[:, i:col_end])
        
        # Rearrange columns
        if mode == "random":
            np.random.shuffle(columns)
        elif mode == "intensity":
            # Sort by average spectral intensity (sum of pixel values)
            intensities = [np.sum(col) for col in columns]
            columns = [col for _, col in sorted(zip(intensities, columns), reverse=True)]
        
        # Reassemble image
        shuffled = np.hstack(columns)
        # Pad or trim to original width if needed
        if shuffled.shape[1] < w:
            shuffled = np.pad(shuffled, ((0, 0), (0, w - shuffled.shape[1])), mode='constant')
        elif shuffled.shape[1] > w:
            shuffled = shuffled[:, :w]
        return shuffled
        # Audio:
        # - Random: Time chunks shuffled chaotically—like a glitchy remix.
        # - Intensity: Orders loudest to quietest (or vice versa)—restructures dynamics dramatically.
    elif effect == "kaleidoscope":
        # Kaleidoscope effect: radial symmetry
        rows, cols = img.shape
        center = (cols // 2, rows // 2)
        # Convert to polar coordinates and repeat sections
        polar = cv2.warpPolar(img, (rows, cols), center, max(rows, cols), cv2.WARP_POLAR_LINEAR)
        kaleido = np.tile(polar[:, :cols // param], (1, param))[:, :cols]
        # Back to cartesian
        kaleido = cv2.warpPolar(kaleido, (rows, cols), center, max(rows, cols), cv2.WARP_INVERSE_MAP)
        return kaleido
        # Audio: Repeats and mirrors time segments radially—echoey, hypnotic loops.

    elif effect == "fractal_noise":
        # Add fractal-like noise pattern
        rows, cols = img.shape
        noise = np.random.random((rows, cols))
        for _ in range(int(param)):
            noise = cv2.pyrDown(noise)  # Downscale
            noise = cv2.pyrUp(noise, dstsize=(rows, cols))  # Upscale back
        fractal = np.clip(img + noise * 50, 0, 255).astype(np.uint8)
        return fractal
        # Audio: Gritty, layered noise texture—like static with depth and shimmer.

    elif effect == "vortex":
        # Spiral vortex distortion
        rows, cols = img.shape
        center = (cols // 2, rows // 2)
        x, y = np.indices((rows, cols))
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        theta = np.arctan2(y - center[1], x - center[0]) + param * r / max(rows, cols)
        x_new = center[0] + r * np.cos(theta)
        y_new = center[1] + r * np.sin(theta)
        vortex = cv2.remap(img, x_new.astype(np.float32), y_new.astype(np.float32), 
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return vortex
        # Audio: Spiraling pitch and time warp—like a sound funneling into itself.

    elif effect == "stretch":
        # Stretch horizontally or vertically
        rows, cols = img.shape
        if param > 0:  # Positive = horizontal stretch
            stretched = cv2.resize(img, (int(cols * param), rows), interpolation=cv2.INTER_LINEAR)
            stretched = cv2.resize(stretched, (cols, rows), interpolation=cv2.INTER_LINEAR)
        else:  # Negative = vertical stretch
            stretched = cv2.resize(img, (cols, int(rows * -param)), interpolation=cv2.INTER_LINEAR)
            stretched = cv2.resize(stretched, (cols, rows), interpolation=cv2.INTER_LINEAR)
        return stretched
        # Audio: 
        # - Horizontal: Slows or speeds time without pitch change.
        # - Vertical: Compresses or expands frequency range.

    elif effect == "erosion":
        # Morphological erosion—shrinks bright areas
        kernel = np.ones((int(param), int(param)), np.uint8)
        eroded = cv2.erode(img, kernel, iterations=1)
        return eroded
        # Audio: Thins out loud parts—hollow, subdued texture, like a quieted echo.
    elif effect == "dilation":
        # Morphological dilation—expands bright areas
        kernel = np.ones((int(param), int(param)), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=1)
        return dilated
        # Audio: Fattens loud parts—thicker, more resonant texture, like a boosted sustain.

    elif effect == "opening":
        # Erosion followed by dilation—removes small bright spots, smooths edges
        kernel = np.ones((int(param), int(param)), np.uint8)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opened
        # Audio: Cleans out fine details, leaving a smoother, muted version—soft and airy.

    elif effect == "closing":
        # Dilation followed by erosion—fills small dark gaps, connects bright areas
        kernel = np.ones((int(param), int(param)), np.uint8)
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return closed
        # Audio: Bridges quiet gaps, creating a more continuous, swollen sound—warm and dense.

    elif effect == "gradient":
        # Morphological gradient—difference between dilation and erosion
        kernel = np.ones((int(param), int(param)), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=1)
        eroded = cv2.erode(img, kernel, iterations=1)
        grad = cv2.subtract(dilated, eroded)
        return np.clip(grad, 0, 255).astype(np.uint8)
        # Audio: Highlights edges of loudness—sharp, outline-like texture, crisp and skeletal.

    elif effect == "top_hat":
        # Difference between original and opening—isolates small bright peaks
        kernel = np.ones((int(param), int(param)), np.uint8)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        tophat = cv2.subtract(img, opened)
        return np.clip(tophat, 0, 255).astype(np.uint8)
        # Audio: Pulls out transient spikes—percussive, staccato bursts stand out.
        
    else:
        raise ValueError("Unsupported effect")

def image_to_audio(img, original_audio, sr, n_fft, hop_length):
    spec_shape = librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length).shape
    img_resized = cv2.resize(img, (spec_shape[1], spec_shape[0]), interpolation=cv2.INTER_LINEAR)
    mag = (img_resized / 255.0) * (librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max).max() - 
                                   librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max).min()) + \
                                   librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max).min()
    mag = librosa.db_to_amplitude(mag)
    _, phase = librosa.magphase(librosa.stft(original_audio, n_fft=n_fft, hop_length=hop_length))
    spec = mag * np.exp(1j * phase)
    audio_out = librosa.istft(spec, hop_length=hop_length, length=len(original_audio))
    # Normalize audio
    audio_out = normalize_audio(audio_out)
    return np.clip(audio_out, -1.0, 1.0)

# Example workflow with new effects
if __name__ == "__main__":
    # Convert audio to image
    img, original_audio, sr, n_fft, hop_length = audio_to_image("sequence.wav", size=256)
    show_image(img, "Original Audio Image")

    # Apply effects
    effects = [
        ("blur", 5, "Blurry Audio Image"),
        ("swirl", 5, "Swirled Audio Image"),
        ("rotate", 45, "Rotated Audio Image"),
        ("pixelate", 8, "Pixelated Audio Image"),    # New: blocky glitch
        ("wave", 10, "Wavy Audio Image"),           # New: sinusoidal warp
        ("edge", 2, "Edged Audio Image"),           # New: sharpens transitions
        ("invert", None, "Inverted Audio Image"),    # New: flips amplitude
        ("ripple", 0.5, "Rippled Audio Image"),      # param: ripple intensity (0.1–1.0)
        ("shear", 0.3, "Sheared Audio Image"),       # param: shear amount (0.1–0.5)
        ("threshold", 0.5, "Thresholded Audio Image"), # param: cutoff (0.0–1.0)
        ("emboss", 1.0, "Embossed Audio Image"),     # param: edge strength (0.5–2.0)
        ("solarize", 0.5, "Solarized Audio Image"),  # param: threshold (0.0–1.0)
        ("posterize", 4, "Posterized Audio Image"),   # param: number of levels (2–8)
        ("mirror", "horizontal", "Horizontal Mirror Audio Image"),
        #("mirror", "vertical", "Vertical Mirror Audio Image"),
        #("mirror", "both", "Diagonal Mirror Audio Image"),
        ("jagged_pixelate", 8, "Jagged Pixelated Audio Image"),
        ("shifted_pixelate", 8, "Shifted Pixelated Audio Image"),
        ("column_shuffle", (8, "random"), "Random Column Shuffle Audio Image"),
        ("column_shuffle", (8, "intensity"), "Intensity Column Shuffle Audio Image"),
        ("kaleidoscope", 4, "Kaleidoscope Audio Image"),    # param: symmetry segments (2–8)
        ("fractal_noise", 3, "Fractal Noise Audio Image"),  # param: noise iterations (1–5)
        ("vortex", 0.5, "Vortex Audio Image"),             # param: spiral strength (0.1–1.0)
        ("stretch", 2.0, "Horizontal Stretch Audio Image"), # param: stretch factor (>0 horizontal, <0 vertical)
        #("stretch", -2.0, "Vertical Stretch Audio Image"),
        ("erosion", 3, "Eroded Audio Image"),               # param: kernel size (2–5)
        ("dilation", 3, "Dilated Audio Image"),    # param: kernel size (2–5)
        ("opening", 3, "Opened Audio Image"),      # param: kernel size (2–5)
        ("closing", 3, "Closed Audio Image"),      # param: kernel size (2–5)
        ("gradient", 3, "Gradient Audio Image"),   # param: kernel size (2–5)
        ("top_hat", 3, "Top Hat Audio Image")      # param: kernel size (2–5)
    ]

    output_files = []

    for effect, param, title in effects:
        effected_img = apply_visual_effects(img, effect=effect, param=param)
        print(title)
        show_image(effected_img, title)
        audio_out = image_to_audio(effected_img, original_audio, sr, n_fft, hop_length)
        output_file = f"{effect}_{param[1] if effect == 'column_shuffle' else param}_output.wav" if effect == "column_shuffle" else f"{effect}_output.wav"
        sf.write(output_file, audio_out, sr, subtype='PCM_16')
        output_files.append(output_file)
        
        

    print("Check out your new audio effects:", ", ".join(output_files))
