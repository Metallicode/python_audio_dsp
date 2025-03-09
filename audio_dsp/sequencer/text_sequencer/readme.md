
## Project Overview
This project consists of two Python scripts designed to work together to analyze and sequence audio samples:

1. **Spectrogram Clustering Script**: Analyzes `.wav` files, converts them to spectrograms, clusters them based on visual similarity, and visualizes the results interactively. It outputs a cluster mapping file for use in sequencing.
2. **Cluster-Based Sequencer Script**: Uses the cluster mapping to generate a `.wav` file where each track in a pattern randomly selects samples from a designated cluster, with customizable timing and output length.

Together, these scripts enable you to:
- Group similar audio samples (e.g., kicks, snares, hi-hats) into clusters.
- Create a rhythmic sequence where samples from the same cluster are interchangeable, adding variety to your patterns.

---

## Script 1: Spectrogram Clustering Script

### Purpose
This script processes a directory of `.wav` files, converts each into a spectrogram, clusters them into `k` groups based on spectrogram similarity using K-Means, and displays an interactive Tkinter map where you can hover over spectrograms to play their audio. It outputs a `cluster_mapping.json` file for the sequencer.

### Requirements
- **Python Libraries**: `numpy`, `librosa`, `matplotlib`, `pillow`, `umap-learn`, `sklearn`, `pygame`, `tkinter`.
- **Input**: A `samples` directory containing `.wav` files (e.g., `samples/RX11 SNR 1.wav`).
- **Output**: 
  - `cluster_mapping.json`: A JSON file mapping cluster IDs to sample paths.
  - Interactive Tkinter window for visualization.

### Parameters
- **`SAMPLE_DIR`**: Directory with `.wav` files (default: `"samples"`).
- **`SAMPLE_RATE`**: Audio sample rate (default: `44100` Hz).
- **`SPECTROGRAM_SIZE`**: Spectrogram image size (default: `(100, 100)` pixels).
- **`NUM_CLUSTERS`**: Number of clusters for K-Means (default: `3`).

### Usage
1. **Setup**:
   - Place your `.wav` files in a `samples` folder in the script’s directory.
   - Install dependencies:
     ```bash
     pip install numpy librosa matplotlib pillow umap-learn scikit-learn pygame
     ```

2. **Run**:
   ```bash
   python clustering_script.py
   ```
   - Replace `clustering_script.py` with your script’s filename.

3. **Output**:
   - **Console**: Prints cluster assignments (e.g., `Cluster 0: [samples/RX11 CLHAT1.wav, ...]`).
   - **JSON File**: `cluster_mapping.json` (e.g., `{"0": ["samples/RX11 CLHAT1.wav", ...], ...}`).
   - **UI**: Tkinter window showing spectrograms scattered by UMAP coordinates. Hover to play samples with cluster info (e.g., `Playing: samples/RX11 SNR 1.wav (Cluster 1)`).

### Example Output
For 27 RX11 samples with `NUM_CLUSTERS = 3`:
```
Cluster Mapping:
Cluster 0:
  - samples/RX11 CLHAT1.wav
  - samples/RX11 SHAKER.wav
  ...
Cluster 1:
  - samples/RX11 SNR 1.wav
  - samples/RX11 BD2.wav
  ...
Cluster 2:
  - samples/RX11 TOM2.wav
  - samples/RX11 SNR 4.wav
  ...
Cluster mapping saved to 'cluster_mapping.json'
```

---

## Script 2: Cluster-Based Sequencer Script

### Purpose
This script reads a pattern file and the `cluster_mapping.json` from the clustering script, assigns each track to a cluster, and generates a `.wav` file where each trigger (`"1"`) in the pattern randomly selects a sample from the track’s cluster. It supports timing modifiers, volume control, and parameterized output length and randomness.

### Requirements
- **Python Libraries**: `numpy`, `librosa`, `soundfile`, `json`, `random`.
- **Input**:
  - `cluster_mapping.json`: Cluster assignments from the clustering script.
  - `pattern.txt`: Text file defining track patterns, modifiers, and volumes.
  - `samples` directory: Same as used in clustering.
- **Output**: `sequence.wav`: The generated audio sequence.

### Parameters
- **`pattern_file`**: Path to pattern file (default: `"pattern.txt"`).
- **`cluster_file`**: Path to cluster mapping JSON (default: `"cluster_mapping.json"`).
- **`samples_dir`**: Directory with `.wav` files (default: `"samples"`).
- **`output_file`**: Output `.wav` file (default: `"sequence.wav"`).
- **`bpm`**: Beats per minute (default: `120`).
- **`output_length`**: Duration of output in seconds (default: `16`).
- **`random_seed`**: Seed for random sample selection (default: `None`; set to an integer like `42` for reproducibility).

### Pattern File Format (`pattern.txt`)
- **Lines**: One per track, followed by a volume list on the last line.
- **Pattern**: Binary string (e.g., `10001000`) where `1` triggers a sample, `0` is silence.
- **Modifiers**: Optional, appended to pattern:
  - `*N`: Speeds up by factor `N` (e.g., `*2` = twice as fast).
  - `/N`: Slows down by factor `N` (e.g., `/2` = half speed).
- **Volumes**: Last line, a Python list (e.g., `[1.0, 0.8, 0.6]`), one per track.

#### Example `pattern.txt`
```
1000100010001010/3  ; Track 1: Cluster 2 (toms/snares), 1/3 speed
00001000            ; Track 2: Cluster 1 (snares/claps), normal speed
100001000010        ; Track 3: Cluster 0 (hi-hats/crashes), normal speed
[1.0, 0.8, 0.6]    ; Volumes for tracks 1, 2, 3
```

### Usage
1. **Setup**:
   - Ensure `cluster_mapping.json` and `samples` folder are ready from the clustering script.
   - Create `pattern.txt` with your patterns.
   - Install dependencies:
     ```bash
     pip install numpy librosa soundfile
     ```

2. **Run**:
   ```bash
   python sequencer_script.py
   ```
   - Replace `sequencer_script.py` with your script’s filename.
   - Customize via arguments:
     ```python
     sequencer(output_length=32, random_seed=123, bpm=140)
     ```

3. **Output**:
   - **Console**: Logs cluster mapping, pattern extensions, and trigger details (e.g., `Trigger at 0.00000s with RX11 TOM2.wav`).
   - **WAV File**: `sequence.wav` with the mixed sequence.

### Example Output
```
Random seed: 42
Loaded clusters: {'2': 5, '1': 14, '0': 8}
Track to cluster mapping: {1: '2', 2: '1', 3: '0'}
Output length: 16s (705600 samples)
Parsed patterns and modifiers:
Track 1 (Cluster 2): 1000100010001010 → ... (modifier: /3, step_duration: 0.37500s)
Track 2 (Cluster 1): 00001000 → ... (modifier: none, step_duration: 0.12500s)
Track 3 (Cluster 0): 100001000010 → ... (modifier: none, step_duration: 0.12500s)
Track 1 (Cluster 2) triggers:
Trigger at 0.00000s with RX11 TOM2.wav
Trigger at 1.50000s with RX11 SNR 4.wav
...
Sequence saved to sequence.wav (duration: 16.00000s)
```

---

## Workflow

### Step 1: Cluster Your Samples
1. Place `.wav` files in `samples/` (e.g., RX11 drum hits).
2. Run the **Spectrogram Clustering Script**:
   - Adjust `NUM_CLUSTERS` if needed (e.g., `5` for finer grouping).
   - Check the console and UI to verify clusters (e.g., kicks in one, snares in another).
3. Output: `cluster_mapping.json` (e.g., `{"0": ["samples/RX11 CLHAT1.wav", ...], ...}`).

### Step 2: Define Your Pattern
1. Create `pattern.txt` with your desired rhythm:
   - One line per track, matching the number of clusters or fewer.
   - Add modifiers (`*N`, `/N`) for timing variations.
   - End with a volume list.
2. Example: 3 tracks for 3 clusters (kick, snare, hi-hat).

### Step 3: Generate the Sequence
1. Run the **Cluster-Based Sequencer Script**:
   - Set `output_length` (e.g., `32` seconds).
   - Set `random_seed` (e.g., `42`) for consistency, or `None` for variety.
   - Adjust `bpm` if desired.
2. Listen to `sequence.wav` and check the console for sample choices.

### Iteration
- **Adjust Clusters**: Rerun clustering with different `NUM_CLUSTERS` if groupings aren’t ideal.
- **Tweak Patterns**: Edit `pattern.txt` for different rhythms or timing.
- **Experiment**: Change `output_length` or `random_seed` for new variations.

---

## Tips
- **Sample Naming**: Keep filenames descriptive (e.g., `RX11 SNR 1.wav`) for easy cluster verification.
- **Cluster Count**: Start with `NUM_CLUSTERS = 3` or `5`; too many clusters may split similar sounds.
- **Pattern Length**: Short patterns repeat to fill `output_length`, so design accordingly.
- **Overlap**: If samples overlap too much, reduce volumes or adjust patterns.

---

## Example Workflow
1. **Samples**: 27 RX11 drum hits in `samples/`.
2. **Clustering**: Run clustering with `NUM_CLUSTERS = 3` → `cluster_mapping.json`.
3. **Pattern**: Write `pattern.txt`:
   ```
   10001000/2
   00100010
   10000100
   [1.0, 0.8, 0.6]
   ```
4. **Sequencing**: Run sequencer with `output_length=16`, `random_seed=42` → `sequence.wav`.

