import timeit

import numpy as np
import torch.nn as nn
from scipy.signal import firwin, lfilter

from torchfx import Wave
from torchfx.filter import DesignableFIR

SAMPLE_RATE = 44100
REP = 50


def create_audio(sample_rate, duration, num_channels):
    signal = np.random.randn(num_channels, int(sample_rate * duration))
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal


def gpu_fir(wave, fchain):
    _ = wave | fchain


def cpu_fir(wave, fchain):
    _ = wave | fchain


def scipy_fir(signal, bs):
    a = [1]
    filtered_signal = lfilter(bs[0], a, signal)
    filtered_signal = lfilter(bs[1], a, filtered_signal)
    filtered_signal = lfilter(bs[2], a, filtered_signal)
    filtered_signal = lfilter(bs[3], a, filtered_signal)
    filtered_signal = lfilter(bs[4], a, filtered_signal)
    return filtered_signal


def start(out_file):
    print("time,channels,gpu,cpu,scipy")
    times = [5, 60, 180, 300, 600]

    for t in times:
        for i in [1, 2, 4, 8, 12]:
            signal = create_audio(SAMPLE_RATE, t, i)

            wave = Wave(signal, SAMPLE_RATE)
            fchain = nn.Sequential(
                DesignableFIR(num_taps=101, cutoff=1000, fs=SAMPLE_RATE),
                DesignableFIR(num_taps=102, cutoff=5000, fs=SAMPLE_RATE),
                DesignableFIR(num_taps=103, cutoff=1500, fs=SAMPLE_RATE),
                DesignableFIR(num_taps=104, cutoff=1800, fs=SAMPLE_RATE),
                DesignableFIR(num_taps=105, cutoff=1850, fs=SAMPLE_RATE),
            )

            for f in fchain:
                f.compute_coefficients()

            wave.to("cuda")
            fchain.to("cuda")
            gpu_fir_time = timeit.timeit(lambda: gpu_fir(wave, fchain), number=REP)

            wave.to("cpu")
            fchain.to("cpu")
            cpu_fir_time = timeit.timeit(lambda: cpu_fir(wave, fchain), number=REP)

            b1 = firwin(101, 1000, fs=SAMPLE_RATE)
            b2 = firwin(102, 5000, fs=SAMPLE_RATE)
            b3 = firwin(103, 1500, fs=SAMPLE_RATE)
            b4 = firwin(104, 1800, fs=SAMPLE_RATE)
            b5 = firwin(105, 1850, fs=SAMPLE_RATE)

            scipy_fir_time = timeit.timeit(
                lambda: scipy_fir(signal, [b1, b2, b3, b4, b5]), number=REP
            )
            print(
                f"{t},{i},{gpu_fir_time/REP:.6f},{cpu_fir_time/REP:.6f},{scipy_fir_time/REP:.6f}",
            )


if __name__ == "__main__":
    with open("fir.out", "w") as f:
        start(f)
