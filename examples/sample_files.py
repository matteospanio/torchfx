import timeit
import numpy as np
from scipy.signal import lfilter, firwin
import torch.nn as nn
from torchfx import Wave
from torchfx.filter import DesignableFIR

SAMPLE_RATE = 44100


def create_audio(sample_rate, duration, num_channels):
    signal = np.random.randn(num_channels, int(sample_rate * duration))
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal


def gpu_fir(signal, sample_rate):
    wave = Wave(signal, sample_rate)
    wave.to("cuda")

    fchain = nn.Sequential(
        DesignableFIR(num_taps=101, cutoff=1000),
        DesignableFIR(num_taps=102, cutoff=5000),
        DesignableFIR(num_taps=103, cutoff=1500),
        DesignableFIR(num_taps=104, cutoff=1800),
    )

    _ = wave | fchain


def cpu_fir(signal, sample_rate):
    wave = Wave(signal, sample_rate)

    fchain = nn.Sequential(
        DesignableFIR(num_taps=101, cutoff=1000),
        DesignableFIR(num_taps=102, cutoff=5000),
        DesignableFIR(num_taps=103, cutoff=1500),
        DesignableFIR(num_taps=104, cutoff=1800),
    )

    _ = wave | fchain


def scipy_fir(signal, sample_rate):
    b1 = firwin(101, 1000, fs=sample_rate)
    b2 = firwin(102, 1000, fs=sample_rate)
    b3 = firwin(103, 1000, fs=sample_rate)
    b4 = firwin(104, 1000, fs=sample_rate)
    a = [1]
    filtered_signal = lfilter(b1, a, signal)
    filtered_signal = lfilter(b2, a, filtered_signal)
    filtered_signal = lfilter(b3, a, filtered_signal)
    filtered_signal = lfilter(b4, a, filtered_signal)
    return filtered_signal


def fir_bench():
    for i in range(1, 8):
        signal = create_audio(SAMPLE_RATE, 60, i)

        gpu_fir_time = timeit.timeit(lambda: gpu_fir(signal, SAMPLE_RATE), number=100)
        cpu_fir_time = timeit.timeit(lambda: cpu_fir(signal, SAMPLE_RATE), number=100)
        scipy_fir_time = timeit.timeit(
            lambda: scipy_fir(signal, SAMPLE_RATE), number=100
        )
        print(
            cpu_fir_time,
            scipy_fir_time,
            f"Channels: {i}, GPU FIR: {gpu_fir_time:.4f}s, CPU FIR: {cpu_fir_time:.4f}s, SciPy FIR: {scipy_fir_time:.4f}s",
        )


if __name__ == "__main__":
    fir_bench()
