"""
Drop-in replacement for scipy.io.wavfile.

Supports reading and writing WAV files with proper chunk parsing
and multiple audio formats (16/24/32-bit PCM, 32/64-bit float).
"""

import numpy as np
import struct
import os


def _find_chunk(f, chunk_id):
    """
    Search for a chunk in a WAV file by its 4-byte ID.

    Args:
        f: File object positioned after RIFF header (at byte 12).
        chunk_id: 4-byte chunk identifier (e.g., b'fmt ', b'data').

    Returns:
        Tuple of (chunk_size, chunk_data_position) or (None, None) if not found.
    """
    while True:
        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            return None, None

        current_id = chunk_header[:4]
        chunk_size = struct.unpack('<I', chunk_header[4:8])[0]

        if current_id == chunk_id:
            return chunk_size, f.tell()

        # Skip to next chunk (chunks are word-aligned)
        skip = chunk_size + (chunk_size % 2)
        f.seek(skip, 1)


def read(filename):
    """
    Drop-in replacement for scipy.io.wavfile.read.

    Properly parses WAV chunk structure to handle files with extended
    headers, metadata, or non-standard chunk ordering.

    Supports:
        - 8-bit unsigned PCM (uint8)
        - 16-bit signed PCM (int16)
        - 24-bit signed PCM (int32, scaled)
        - 32-bit signed PCM (int32)
        - 32-bit IEEE float (float32)
        - 64-bit IEEE float (float64)

    Args:
        filename (str): Path to the WAV file.

    Returns:
        rate (int): Sample rate of the file.
        data (numpy.ndarray): Audio data as numpy array.
                              Shape is (N,) for mono or (N, channels) for multi-channel.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, 'rb') as f:
        # Validate RIFF header
        riff = f.read(4)
        if riff != b'RIFF':
            raise ValueError(f"Not a valid WAV file: missing RIFF header")

        f.read(4)  # Skip file size

        wave = f.read(4)
        if wave != b'WAVE':
            raise ValueError(f"Not a valid WAV file: missing WAVE format")

        # Find and parse 'fmt ' chunk
        fmt_size, fmt_pos = _find_chunk(f, b'fmt ')
        if fmt_size is None:
            raise ValueError("Invalid WAV file: missing 'fmt ' chunk")

        fmt_data = f.read(fmt_size)

        audio_format = struct.unpack('<H', fmt_data[0:2])[0]
        channels = struct.unpack('<H', fmt_data[2:4])[0]
        rate = struct.unpack('<I', fmt_data[4:8])[0]
        # bytes 8-12: byte rate (skip)
        # bytes 12-14: block align (skip)
        bits_per_sample = struct.unpack('<H', fmt_data[14:16])[0]

        # audio_format: 1 = PCM, 3 = IEEE float, 0xFFFE = extensible
        if audio_format == 0xFFFE and fmt_size >= 40:
            # Extensible format - real format is in sub-format GUID
            # Bytes 24-26 contain the actual format code
            audio_format = struct.unpack('<H', fmt_data[24:26])[0]

        # Seek back to after WAVE header to find data chunk
        f.seek(12)
        data_size, data_pos = _find_chunk(f, b'data')
        if data_size is None:
            raise ValueError("Invalid WAV file: missing 'data' chunk")

        # Read audio data
        raw_data = f.read(data_size)

    # Convert to numpy array based on format
    if audio_format == 1:  # PCM
        if bits_per_sample == 8:
            data = np.frombuffer(raw_data, dtype=np.uint8)
        elif bits_per_sample == 16:
            data = np.frombuffer(raw_data, dtype=np.int16)
        elif bits_per_sample == 24:
            # 24-bit needs special handling - pack into int32
            n_samples = len(raw_data) // 3
            data = np.zeros(n_samples, dtype=np.int32)
            for i in range(n_samples):
                # Little-endian 24-bit to int32 with sign extension
                b = raw_data[i*3:(i+1)*3]
                val = b[0] | (b[1] << 8) | (b[2] << 16)
                if val >= 0x800000:  # Sign extend
                    val -= 0x1000000
                data[i] = val
        elif bits_per_sample == 32:
            data = np.frombuffer(raw_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported PCM bit depth: {bits_per_sample}")

    elif audio_format == 3:  # IEEE float
        if bits_per_sample == 32:
            data = np.frombuffer(raw_data, dtype=np.float32)
        elif bits_per_sample == 64:
            data = np.frombuffer(raw_data, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported float bit depth: {bits_per_sample}")
    else:
        raise ValueError(f"Unsupported audio format: {audio_format}")

    # Reshape for multi-channel
    if channels > 1:
        num_frames = len(data) // channels
        data = data[:num_frames * channels].reshape(-1, channels)

    return rate, data


def write(filename, rate, data):
    """
    Drop-in replacement for scipy.io.wavfile.write.

    Automatically determines output format based on input dtype:
        - int16 input  -> 16-bit PCM output
        - int32 input  -> 32-bit PCM output
        - float32 input -> 32-bit float output
        - float64 input -> 64-bit float output
        - Other int types -> 16-bit PCM (with conversion)

    Float data in [-1.0, 1.0] range is written as IEEE float (not converted to int).
    This matches scipy.io.wavfile behavior.

    Args:
        filename (str): Path to output file.
        rate (int): Sample rate (e.g., 44100).
        data (numpy.ndarray): Audio data. Shape (N,) for mono or (N, channels) for multi-channel.
    """
    data = np.asarray(data)

    # Determine format based on dtype
    if data.dtype == np.int16:
        audio_format = 1  # PCM
        bits_per_sample = 16
        out_data = data
    elif data.dtype == np.int32:
        audio_format = 1  # PCM
        bits_per_sample = 32
        out_data = data
    elif data.dtype == np.float32:
        audio_format = 3  # IEEE float
        bits_per_sample = 32
        out_data = data
    elif data.dtype == np.float64:
        audio_format = 3  # IEEE float
        bits_per_sample = 64
        out_data = data
    elif data.dtype == np.uint8:
        audio_format = 1  # PCM
        bits_per_sample = 8
        out_data = data
    elif data.dtype.kind == 'f':
        # Other float types -> float32
        audio_format = 3
        bits_per_sample = 32
        out_data = data.astype(np.float32)
    elif data.dtype.kind in ('i', 'u'):
        # Other int types -> int16
        audio_format = 1
        bits_per_sample = 16
        out_data = data.astype(np.int16)
    else:
        raise ValueError(f"Unsupported dtype: {data.dtype}")

    # Handle channels
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]

    # Calculate header values
    bytes_per_sample = bits_per_sample // 8
    byte_rate = rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    data_bytes = out_data.tobytes()

    # Build WAV header
    header = b'RIFF'
    header += struct.pack('<I', 36 + len(data_bytes))  # File size - 8
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)                    # fmt chunk size
    header += struct.pack('<H', audio_format)          # Audio format (1=PCM, 3=float)
    header += struct.pack('<H', channels)              # Number of channels
    header += struct.pack('<I', rate)                  # Sample rate
    header += struct.pack('<I', byte_rate)             # Byte rate
    header += struct.pack('<H', block_align)           # Block align
    header += struct.pack('<H', bits_per_sample)       # Bits per sample
    header += b'data'
    header += struct.pack('<I', len(data_bytes))       # Data chunk size

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data_bytes)
