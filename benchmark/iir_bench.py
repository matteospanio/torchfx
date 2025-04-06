import timeit
import numpy as np
from scipy.signal import butter, cheby1, lfilter
import torch.nn as nn
from torchfx import Wave
from torchfx.filter import HiButterworth, LoButterworth, HiChebyshev1, LoChebyshev1

SAMPLE_RATE = 44100


def create_audio(sample_rate, duration, num_channels):
    signal = np.random.randn(num_channels, int(sample_rate * duration))
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal


def gpu_filter(wave, fchain):
    _ = wave | fchain


def cpu_filter(wave, fchain):
    _ = wave | fchain


def scipy_filter(signal, bs, as_):
    filtered_signal = lfilter(bs[0], as_[0], signal)
    filtered_signal = lfilter(bs[1], as_[1], filtered_signal)
    filtered_signal = lfilter(bs[2], as_[2], filtered_signal)
    filtered_signal = lfilter(bs[3], as_[3], filtered_signal)
    return filtered_signal


def filter_bench():
    times = [1]
    for i in range(60, 601, 60):
        times.append(i)

    for t in times:
        for i in range(1, 13):
            signal = create_audio(SAMPLE_RATE, t, i)

            wave = Wave(signal, SAMPLE_RATE)
            fchain = nn.Sequential(
                HiButterworth(cutoff=1000, order=2, fs=SAMPLE_RATE),
                LoButterworth(cutoff=5000, order=2, fs=SAMPLE_RATE),
                HiChebyshev1(cutoff=1500, order=2, fs=SAMPLE_RATE),
                LoChebyshev1(cutoff=1800, order=2, fs=SAMPLE_RATE),
            )

            for f in fchain:
                f.compute_coefficients()

            wave.to("cuda")
            fchain.to("cuda")
            gpu_filter_time = timeit.timeit(lambda: gpu_filter(wave, fchain), number=50)

            wave.to("cpu")
            fchain.to("cpu")
            cpu_filter_time = timeit.timeit(lambda: cpu_filter(wave, fchain), number=50)

            # SciPy filter coefficients
            b1, a1 = butter(2, 1000, btype="high", fs=SAMPLE_RATE)  # type: ignore
            b2, a2 = butter(2, 5000, btype="low", fs=SAMPLE_RATE)  # type: ignore
            b3, a3 = cheby1(2, 0.5, 1500, btype="high", fs=SAMPLE_RATE)  # type: ignore
            b4, a4 = cheby1(2, 0.5, 1800, btype="low", fs=SAMPLE_RATE)  # type: ignore

            scipy_filter_time = timeit.timeit(
                lambda: scipy_filter(signal, [b1, b2, b3, b4], [a1, a2, a3, a4]),
                number=50,
            )

            print(f"Times: {t}\tChannels:{i}")
            print(
                f"GPU:\t{gpu_filter_time:.6f}s, CPU:\t{cpu_filter_time:.6f}s, SciPy:\t{scipy_filter_time:.6f}s",
            )


if __name__ == "__main__":
    filter_bench()
