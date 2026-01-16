import numpy as np
from audio_dsp.utils import wav_io as wavfile
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

class TreeNode:
    def __init__(self, freq, duration, level, angle=0, parent=None):
        self.freq = freq
        self.duration = duration
        self.level = level
        self.angle = angle
        self.parent = parent
        self.children = []
        self.x = 0  # For visualization
        self.y = -level  # Y decreases with level

def build_tree(root_freq=261.63, root_duration=1.0, n_splits=3, angle=30.0, levels=3, speed_factor=1.0):
    """Build a rule-based tree with frequency and duration nodes, adjustable speed."""
    root = TreeNode(root_freq, root_duration * speed_factor, 0)
    nodes = [root]
    x_offset = 0
    
    for level in range(1, levels):
        new_nodes = []
        level_width = len([n for n in nodes if n.level == level - 1]) * n_splits
        x_step = 2.0 / (level_width if level_width > 0 else 1)  # Spread horizontally
        
        for parent in [n for n in nodes if n.level == level - 1]:
            for i in range(n_splits):
                split_angle = angle * (i - (n_splits - 1) / 2)
                child_freq = parent.freq * (2 ** (split_angle / 360))
                # Logarithmic duration with speed factor
                child_duration = root_duration * speed_factor * (2 ** (-level / 2))
                child = TreeNode(child_freq, child_duration, level, split_angle, parent)
                child.x = parent.x + (i - (n_splits - 1) / 2) * x_step
                parent.children.append(child)
                new_nodes.append(child)
            x_offset += x_step * n_splits
        nodes.extend(new_nodes)
    
    # Log tree structure
    for node in nodes:
        print(f"Level {node.level}, Freq: {node.freq:.2f} Hz, Duration: {node.duration:.3f}s, Angle: {node.angle:.1f}°")
    return root, nodes

def traverse_tree(root, method='depth_first'):
    """Traverse the tree and collect frequencies and durations."""
    path = []
    
    def depth_first(node):
        path.append((node.freq, node.duration))
        for child in node.children:
            depth_first(child)
    
    def breadth_first(root):
        queue = [root]
        while queue:
            node = queue.pop(0)
            path.append((node.freq, node.duration))
            queue.extend(node.children)
    
    if method == 'depth_first':
        depth_first(root)
    elif method == 'breadth_first':
        breadth_first(root)
    else:
        raise ValueError("Traversal method must be 'depth_first' or 'breadth_first'")
    
    print(f"Traversal path ({method}): {[(f'{f:.2f} Hz', f'{d:.3f}s') for f, d in path[:5]]} ...")
    return path

def generate_audio_from_tree(path, sample_rate=44100, overlap=0.1, speed_factor=1.0):
    """Generate audio by traversing the tree path with crossfade, adjustable speed."""
    overlap = overlap * speed_factor  # Scale overlap with speed
    total_duration = sum(d for _, d in path) + overlap * (len(path) - 1)
    total_samples = int(total_duration * sample_rate)
    output = np.zeros(total_samples, dtype=np.float64)
    t_offset = 0
    last_phase = 0
    
    for i, (freq, duration) in enumerate(path):
        samples = int(duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        t = np.linspace(0, duration, samples, endpoint=False)
        # Phase continuity
        phase = 2 * np.pi * freq * t + last_phase
        tone = np.sin(phase)
        last_phase = phase[-1] % (2 * np.pi)
        
        # Crossfade
        start = int(t_offset * sample_rate)
        end = start + samples
        fade_in = np.linspace(0, 1, overlap_samples) if i > 0 else np.ones(overlap_samples)
        fade_out = np.linspace(1, 0, overlap_samples) if i < len(path) - 1 else np.ones(overlap_samples)
        
        if i > 0:
            output[start:start + overlap_samples] *= fade_out
            output[start:start + overlap_samples] += tone[:overlap_samples] * fade_in
        output[start + overlap_samples:end] += tone[overlap_samples:]
        
        t_offset += duration
        print(f"Node {i}: Freq: {freq:.2f} Hz, Duration: {duration:.3f}s, Start: {start}, Overlap: {overlap_samples}")
    
    # Normalize
    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Total duration: {total_duration:.2f}s")
    return output, sample_rate

def visualize_tree(nodes):
    """Visualize the tree structure."""
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = []
    for node in nodes:
        if node.parent:
            lines.append([(node.parent.x, node.parent.y), (node.x, node.y)])
    lc = LineCollection(lines, colors='b', linewidths=1)
    ax.add_collection(lc)
    
    for node in nodes:
        ax.plot(node.x, node.y, 'o', color='r' if node.level == 0 else 'g' if not node.children else 'b')
        ax.text(node.x, node.y + 0.1, f"{node.freq:.0f}\n{node.duration:.2f}", ha='center', fontsize=8)
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-levels - 0.5, 0.5)
    ax.set_title("Rule-Based Tree Structure")
    ax.set_xlabel("Angle Spread")
    ax.set_ylabel("Level (Root to Leaves)")
    plt.show()

def rule_based_tree_composer(root_freq=261.63, root_duration=1.0, n_splits=3, angle=30.0, levels=3, traversal='depth_first', sample_rate=44100, overlap=0.1, speed_factor=1.0, visualize=False):
    """Generate a musical composition from a rule-based tree with speed control."""
    # Build the tree
    root, nodes = build_tree(root_freq, root_duration, n_splits, angle, levels, speed_factor)
    
    # Visualize the tree
    if visualize:
        visualize_tree(nodes)
    
    # Traverse the tree
    path = traverse_tree(root, method=traversal)
    
    # Generate audio
    audio, sr = generate_audio_from_tree(path, sample_rate, overlap, speed_factor)
    
    # Waveform and spectrogram visualization
    if visualize:
        times = np.linspace(0, len(audio) / sr, len(audio))
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(times, audio, label="Tree Composition", color='r', alpha=0.5)
        plt.title(f"Waveform of Rule-Based Tree Composer ({traversal}, Speed: {speed_factor}x)")
        plt.subplot(2, 1, 2)
        plt.specgram(audio, Fs=sr, NFFT=2048, noverlap=512, mode='magnitude', scale='linear')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title("Spectrogram of Tree Composition")
        plt.tight_layout()
        plt.show()
    
    return audio

# Test it
if __name__ == "__main__":
    # Define parameters
    root_freq = 261.63  # C4
    root_duration = 1.0  # 1s root tone base
    n_splits = 5  # 3 branches per split
    angle = 44.0  # 30° angle per split
    levels = 5  # 3 levels (root + 2 splits)
    traversal = 'depth_first'  # Try 'breadth_first'
    speed_factor = 0.3  # Faster—try 2.0 for slower
    
    # Generate Rule-Based Tree Composer with visualization
    audio = rule_based_tree_composer(root_freq, root_duration, n_splits, angle, levels, traversal, speed_factor=speed_factor, visualize=True)
    
    # Save output
    wavfile.write("tree_composer.wav", 44100, audio.astype(np.float32))
    print(f"Rule-Based Tree Composer audio saved to tree_composer.wav")