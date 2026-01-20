"""
Rule-based tree composer.

Generates musical compositions by traversing a tree structure where each node
represents a frequency/duration pair. Supports depth-first and breadth-first
traversals for different musical textures.

This module provides both the legacy functional API and the new unified
class-based API via TreeSequencer.
"""

import numpy as np

try:
    from audio_dsp.utils import wav_io as wavfile
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Import unified base classes
try:
    from .base import GenerativeSequencer
    _HAS_UNIFIED_API = True
except ImportError:
    _HAS_UNIFIED_API = False


class TreeNode:
    """Node in the composition tree."""

    def __init__(self, freq, duration, level, angle=0, parent=None):
        self.freq = freq
        self.duration = duration
        self.level = level
        self.angle = angle
        self.parent = parent
        self.children = []
        self.x = 0
        self.y = -level


# =============================================================================
# Unified API (new)
# =============================================================================

if _HAS_UNIFIED_API:
    class TreeSequencer(GenerativeSequencer):
        """
        Rule-based tree composition sequencer.

        Generates music by building and traversing a tree structure where
        branching creates related frequencies and durations.

        Parameters
        ----------
        sample_rate : int
            Audio sample rate (default: 44100)
        bpm : float
            Tempo in BPM (default: 120)
        root_freq : float
            Root frequency in Hz (default: 261.63, C4)

        Examples
        --------
        >>> seq = TreeSequencer(root_freq=261.63)
        >>> audio = seq.generate_tree_composition(levels=4, n_splits=3, angle=30.0)
        """

        def __init__(self, sample_rate: int = 44100, bpm: float = 120,
                     root_freq: float = 261.63):
            super().__init__(sample_rate, bpm)
            self.root_freq = root_freq
            self._nodes = []
            self._root = None

        def build_tree(self, n_splits: int = 3, angle: float = 30.0,
                       levels: int = 3, speed_factor: float = 1.0):
            """
            Build a rule-based tree with frequency and duration nodes.

            Parameters
            ----------
            n_splits : int
                Number of branches per node (default: 3)
            angle : float
                Angle spread per split in degrees (default: 30.0)
            levels : int
                Number of levels in the tree (default: 3)
            speed_factor : float
                Duration scaling factor (default: 1.0)

            Returns
            -------
            tuple
                (root_node, all_nodes)
            """
            root_duration = self.beat_duration
            root = TreeNode(self.root_freq, root_duration * speed_factor, 0)
            nodes = [root]
            x_offset = 0

            for level in range(1, levels):
                new_nodes = []
                level_width = len([n for n in nodes if n.level == level - 1]) * n_splits
                x_step = 2.0 / (level_width if level_width > 0 else 1)

                for parent in [n for n in nodes if n.level == level - 1]:
                    for i in range(n_splits):
                        split_angle = angle * (i - (n_splits - 1) / 2)
                        child_freq = parent.freq * (2 ** (split_angle / 360))
                        child_duration = root_duration * speed_factor * (2 ** (-level / 2))
                        child = TreeNode(child_freq, child_duration, level, split_angle, parent)
                        child.x = parent.x + (i - (n_splits - 1) / 2) * x_step
                        parent.children.append(child)
                        new_nodes.append(child)
                    x_offset += x_step * n_splits
                nodes.extend(new_nodes)

            self._root = root
            self._nodes = nodes
            return root, nodes

        def traverse_tree(self, method: str = 'depth_first') -> list:
            """
            Traverse the tree and collect frequencies and durations.

            Parameters
            ----------
            method : str
                'depth_first' or 'breadth_first'

            Returns
            -------
            list
                List of (frequency, duration) tuples
            """
            if self._root is None:
                raise ValueError("Tree not built. Call build_tree() first.")

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
                depth_first(self._root)
            elif method == 'breadth_first':
                breadth_first(self._root)
            else:
                raise ValueError("Method must be 'depth_first' or 'breadth_first'")

            return path

        def generate(self, duration: float) -> np.ndarray:
            """
            Generate audio from tree traversal.

            Parameters
            ----------
            duration : float
                Target duration in seconds (tree is repeated as needed)

            Returns
            -------
            np.ndarray
                Generated audio
            """
            if self._root is None:
                self.build_tree()

            path = self.traverse_tree()
            return self.generate_from_path(path, duration)

        def generate_tree_composition(self, levels: int = 3, n_splits: int = 3,
                                      angle: float = 30.0, traversal: str = 'depth_first',
                                      speed_factor: float = 1.0,
                                      overlap: float = 0.1) -> np.ndarray:
            """
            Generate a complete tree-based composition.

            Parameters
            ----------
            levels : int
                Number of tree levels
            n_splits : int
                Branches per node
            angle : float
                Angle spread in degrees
            traversal : str
                'depth_first' or 'breadth_first'
            speed_factor : float
                Duration scaling
            overlap : float
                Crossfade overlap in seconds

            Returns
            -------
            np.ndarray
                Generated audio
            """
            self.build_tree(n_splits, angle, levels, speed_factor)
            path = self.traverse_tree(traversal)
            return self._generate_with_crossfade(path, overlap, speed_factor)

        def generate_from_path(self, path: list, duration: float) -> np.ndarray:
            """Generate audio from a path of (freq, duration) pairs."""
            total_samples = int(duration * self.sample_rate)
            output = np.zeros(total_samples, dtype=np.float32)

            current_pos = 0
            path_idx = 0

            while current_pos < total_samples:
                freq, dur = path[path_idx % len(path)]
                tone = self.generate_tone(freq, dur)
                tone_len = len(tone)
                end_pos = min(current_pos + tone_len, total_samples)
                output[current_pos:end_pos] += tone[:end_pos - current_pos]
                current_pos += tone_len
                path_idx += 1

            return self._normalize(output)

        def _generate_with_crossfade(self, path: list, overlap: float,
                                     speed_factor: float) -> np.ndarray:
            """Generate audio with crossfade between notes."""
            overlap = overlap * speed_factor
            total_duration = sum(d for _, d in path) + overlap * (len(path) - 1)
            total_samples = int(total_duration * self.sample_rate)
            output = np.zeros(total_samples, dtype=np.float64)

            t_offset = 0
            last_phase = 0

            for i, (freq, duration) in enumerate(path):
                samples = int(duration * self.sample_rate)
                overlap_samples = int(overlap * self.sample_rate)
                t = np.linspace(0, duration, samples, endpoint=False)

                phase = 2 * np.pi * freq * t + last_phase
                tone = np.sin(phase)
                last_phase = phase[-1] % (2 * np.pi)

                start = int(t_offset * self.sample_rate)
                end = start + samples
                fade_in = np.linspace(0, 1, overlap_samples) if i > 0 else np.ones(overlap_samples)
                fade_out = np.linspace(1, 0, overlap_samples) if i < len(path) - 1 else np.ones(overlap_samples)

                if i > 0 and start + overlap_samples <= len(output):
                    output[start:start + overlap_samples] *= fade_out
                    output[start:start + overlap_samples] += tone[:overlap_samples] * fade_in
                if start + overlap_samples < end and end <= len(output):
                    output[start + overlap_samples:end] += tone[overlap_samples:]

                t_offset += duration

            return self._normalize(output.astype(np.float32))


# =============================================================================
# Legacy API (backwards compatible)
# =============================================================================

def build_tree(root_freq=261.63, root_duration=1.0, n_splits=3, angle=30.0, levels=3, speed_factor=1.0):
    """Build a rule-based tree with frequency and duration nodes."""
    root = TreeNode(root_freq, root_duration * speed_factor, 0)
    nodes = [root]
    x_offset = 0

    for level in range(1, levels):
        new_nodes = []
        level_width = len([n for n in nodes if n.level == level - 1]) * n_splits
        x_step = 2.0 / (level_width if level_width > 0 else 1)

        for parent in [n for n in nodes if n.level == level - 1]:
            for i in range(n_splits):
                split_angle = angle * (i - (n_splits - 1) / 2)
                child_freq = parent.freq * (2 ** (split_angle / 360))
                child_duration = root_duration * speed_factor * (2 ** (-level / 2))
                child = TreeNode(child_freq, child_duration, level, split_angle, parent)
                child.x = parent.x + (i - (n_splits - 1) / 2) * x_step
                parent.children.append(child)
                new_nodes.append(child)
            x_offset += x_step * n_splits
        nodes.extend(new_nodes)

    for node in nodes:
        print(f"Level {node.level}, Freq: {node.freq:.2f} Hz, Duration: {node.duration:.3f}s, Angle: {node.angle:.1f}")
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
    """Generate audio by traversing the tree path with crossfade."""
    overlap = overlap * speed_factor
    total_duration = sum(d for _, d in path) + overlap * (len(path) - 1)
    total_samples = int(total_duration * sample_rate)
    output = np.zeros(total_samples, dtype=np.float64)
    t_offset = 0
    last_phase = 0

    for i, (freq, duration) in enumerate(path):
        samples = int(duration * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        t = np.linspace(0, duration, samples, endpoint=False)
        phase = 2 * np.pi * freq * t + last_phase
        tone = np.sin(phase)
        last_phase = phase[-1] % (2 * np.pi)

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

    max_amp = np.max(np.abs(output))
    if max_amp > 0:
        output = output / max_amp
    print(f"Output range: {np.min(output):.3f} to {np.max(output):.3f}, Total duration: {total_duration:.2f}s")
    return output, sample_rate


def visualize_tree(nodes, levels=3):
    """Visualize the tree structure."""
    if not _HAS_MATPLOTLIB:
        print("matplotlib not available for visualization")
        return

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


def rule_based_tree_composer(root_freq=261.63, root_duration=1.0, n_splits=3, angle=30.0,
                             levels=3, traversal='depth_first', sample_rate=44100,
                             overlap=0.1, speed_factor=1.0, visualize=False):
    """Generate a musical composition from a rule-based tree."""
    root, nodes = build_tree(root_freq, root_duration, n_splits, angle, levels, speed_factor)

    if visualize and _HAS_MATPLOTLIB:
        visualize_tree(nodes, levels)

    path = traverse_tree(root, method=traversal)
    audio, sr = generate_audio_from_tree(path, sample_rate, overlap, speed_factor)

    if visualize and _HAS_MATPLOTLIB:
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


if __name__ == "__main__":
    root_freq = 261.63
    root_duration = 1.0
    n_splits = 5
    angle = 44.0
    levels = 5
    traversal = 'depth_first'
    speed_factor = 0.3

    audio = rule_based_tree_composer(root_freq, root_duration, n_splits, angle, levels,
                                     traversal, speed_factor=speed_factor, visualize=True)

    wavfile.write("tree_composer.wav", 44100, audio.astype(np.float32))
    print(f"Rule-Based Tree Composer audio saved to tree_composer.wav")
