"""Stream processing example for large audio files.

Demonstrates chunk-based processing of audio files that may be too
large to load into memory entirely.

Usage:
    python stream_processing.py input.wav output.wav
"""

import sys

from torchfx.effect import Gain, Normalize
from torchfx.filter.iir import HiButterworth, LoButterworth
from torchfx.realtime import StreamProcessor

# Build effect chain
# Note: filter cutoffs must be below the Nyquist frequency (fs/2) of the input file.
# Using 6000 Hz is safe for all common sample rates (16kHz+).
effects = [
    HiButterworth(60),           # Remove sub-bass rumble
    LoButterworth(6000),         # Remove high frequencies
    Gain(0.8),                   # Reduce volume slightly
    Normalize(peak=0.95),        # Normalize to -0.5dBFS
]

if len(sys.argv) < 3:
    print("Usage: python stream_processing.py <input.wav> <output.wav>")
    print()
    print("Example using the generator API:")
    print("  Yields processed chunks as tensors for custom pipelines.")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Process file using context manager
with StreamProcessor(effects=effects, chunk_size=65536) as processor:
    print(f"Processing: {input_path} -> {output_path}")
    print(f"Chunk size: {processor.chunk_size} samples")
    print(f"Effects: {len(processor.effects)} in chain")
    processor.process_file(input_path, output_path)

print("Done!")
