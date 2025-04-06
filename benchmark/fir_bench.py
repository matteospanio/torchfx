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


def start():
    times = [1]
    for i in range(60, 601, 60):
        times.append(i)

    for t in times:
        for i in range(1, 9):
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
            gpu_fir_time = timeit.timeit(lambda: gpu_fir(wave, fchain), number=50)

            wave.to("cpu")
            fchain.to("cpu")
            cpu_fir_time = timeit.timeit(lambda: cpu_fir(wave, fchain), number=50)

            b1 = firwin(101, 1000, fs=SAMPLE_RATE)
            b2 = firwin(102, 5000, fs=SAMPLE_RATE)
            b3 = firwin(103, 1500, fs=SAMPLE_RATE)
            b4 = firwin(104, 1800, fs=SAMPLE_RATE)
            b5 = firwin(105, 1850, fs=SAMPLE_RATE)

            scipy_fir_time = timeit.timeit(
                lambda: scipy_fir(signal, [b1, b2, b3, b4, b5]), number=50
            )
            print(f"Times: {t}\tChannels:{i}")
            print(
                f"GPU: {gpu_fir_time/50:.6f}s\tCPU: {cpu_fir_time/50:.6f}s\tSciPy: {scipy_fir_time/50:.6f}s",
            )


if __name__ == "__main__":
    start()
