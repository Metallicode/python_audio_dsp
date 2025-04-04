import numpy as np
import os
import cv2
from pydub import AudioSegment
import subprocess
import shutil

# Game of Life rules
def apply_rules(grid):
    rows, cols = grid.shape
    new_grid = np.zeros_like(grid)
    for i in range(rows):
        for j in range(cols):
            neighbors = np.sum(grid[max(0, i-1):min(rows, i+2), max(0, j-1):min(cols, j+2)]) - grid[i, j]
            if grid[i, j] == 1 and (neighbors == 2 or neighbors == 3):
                new_grid[i, j] = 1
            elif grid[i, j] == 0 and neighbors == 3:
                new_grid[i, j] = 1
    return new_grid

# Add initial patterns
def add_pattern(grid, pattern, x, y):
    pattern = np.array(pattern)
    ph, pw = pattern.shape
    grid[x:x+ph, y:y+pw] = pattern

# Generate a sine wave tone
def generate_tone(frequency, duration_ms, n_cells, sample_rate=44100):
    t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
    tone = np.sin(2 * np.pi * frequency * t)/(n_cells*4)
    # Normalize to 16-bit range
    audio = (tone * 32767).astype(np.int16)
    return AudioSegment(
        audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1
    )

def edo_tone_sequencer(output_video, output_audio, rows=10, cols=10, tempo=120, max_steps=100, init_alive=0.2, base_freq=261.63):
    # Create grid
    grid = np.zeros((rows, cols), dtype=int)
    total_cells = rows * cols
    edo = total_cells  # EDO scale based on number of cells
    
    # Assign frequencies to cells (base_freq = C4, 261.63 Hz)
    freq_grid = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            # Map cell index to EDO step (0 to total_cells-1)
            cell_idx = i * cols + j
            # Frequency = base_freq * 2^(n/EDO), where n is the step in the EDO scale
            freq_grid[i, j] = base_freq * (2 ** (cell_idx / edo))
    
    # Initialize grid
    grid = (np.random.random((rows, cols)) < init_alive).astype(int)
    blinker = [[1, 1, 1]]
    glider = [[0, 1, 0], [0, 0, 1], [1, 1, 1]]
    if rows >= 3 and cols >= 3:
        add_pattern(grid, blinker, 1, 1)
        add_pattern(grid, glider, rows-3, cols-3)
    
    # Audio and video parameters
    step_duration = 60000 / tempo  # ms per step
    fps = 1000 / step_duration     # FPS derived from tempo
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    frames = []
    
    # Audio output
    full_audio = AudioSegment.silent(duration=0)
    
    # Simulation
    step = 0
    prev_grid = None
    while step < max_steps:
        if prev_grid is not None and np.array_equal(grid, prev_grid):
            break
        prev_grid = grid.copy()
        
        # Generate audio for this step
        step_audio = AudioSegment.silent(duration=step_duration)
        live_cells = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i, j] == 1:
                    live_cells += 1
                    freq = freq_grid[i, j]
                    print(f"Step {step}: Triggering tone at row {i}, col {j}, freq {freq:.2f} Hz")
                    tone = generate_tone(freq, step_duration, n_cells=live_cells)
                    step_audio = step_audio.overlay(tone, position=0)
        
        print(f"Step {step}: {live_cells} live cells triggered")
        full_audio += step_audio
        
        # Generate and save video frame
        frame = np.zeros((rows*50, cols*50, 3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                color = (255, 255, 255) if grid[i, j] == 1 else (0, 0, 0)
                cv2.rectangle(frame, (j*50, i*50), ((j+1)*50, (i+1)*50), color, -1)
        frame_path = os.path.join(temp_dir, f"frame_{step:05d}.png")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        
        # Update grid
        grid = apply_rules(grid)
        step += 1
    
    # Export audio as WAV
    print(f"Exporting audio to {output_audio}, duration: {len(full_audio)} ms")
    full_audio.export(output_audio, format="wav")
    
    # Use FFmpeg to create video with audio
    expected_duration = step * step_duration / 1000  # seconds
    print(f"Expected duration: {expected_duration} seconds with {step} steps at {fps:.2f} FPS (tempo={tempo} BPM)")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate", f"{fps:.2f}",
        "-i", os.path.join(temp_dir, "frame_%05d.png"),
        "-i", output_audio,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", "44100",
        "-shortest",
        "-loglevel", "verbose",
        output_video
    ]
    print("Running FFmpeg command:", " ".join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Clean up temporary frames
    shutil.rmtree(temp_dir)
    
    print(f"Simulation completed after {step} steps. Video with audio saved to {output_video}, audio to {output_audio}")

if __name__ == "__main__":
    edo_tone_sequencer(
        output_video="edo_tone_sequencer.mp4",
        output_audio="edo_tone_sequencer.wav",
        rows=10,
        cols=10,
        tempo=120,
        max_steps=100,
        init_alive=0.2,
        base_freq=261.63  # Middle C
    )