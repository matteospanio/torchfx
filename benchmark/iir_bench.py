import timeit

import numpy as np
import torch.nn as nn
from scipy.signal import butter, cheby1, lfilter

from torchfx import Wave
from torchfx.filter import HiButterworth, HiChebyshev1, LoButterworth, LoChebyshev1

SAMPLE_RATE = 44100
REP = 50


def create_audio(sample_rate, duration, num_channels):
    signal = np.random.randn(num_channels, int(sample_rate * duration))
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal


def gpu_filter(wave, fchain):
    _ = fchain(wave.ys)


def cpu_filter(wave, fchain):
    _ = fchain(wave.ys)


def scipy_filter(signal, bs, as_):
    filtered_signal = lfilter(bs[0], as_[0], signal)
    filtered_signal = lfilter(bs[1], as_[1], filtered_signal)
    filtered_signal = lfilter(bs[2], as_[2], filtered_signal)
    filtered_signal = lfilter(bs[3], as_[3], filtered_signal)
    return filtered_signal


def start(outfile):
    times = [1, 5, 180, 300, 600]

    print("time,channels,gpu,cpu,scipy")
    for t in times:
        for i in [1, 2, 4, 8, 12]:
            signal = create_audio(SAMPLE_RATE, t, i)

            wave = Wave(signal, SAMPLE_RATE)
            fchain = nn.Sequential(
                HiButterworth(cutoff=1000, order=2, fs=SAMPLE_RATE),
                LoButterworth(cutoff=5000, order=2, fs=SAMPLE_RATE),
                HiChebyshev1(cutoff=1500, order=2, fs=SAMPLE_RATE),
                LoChebyshev1(cutoff=1800, order=2, fs=SAMPLE_RATE),
            )

            wave.to("cuda")
            fchain.to("cuda")

            for f in fchain:
                f.compute_coefficients()
                f.move_coeff("cuda")

            gpu_filter_time = timeit.timeit(lambda: gpu_filter(wave, fchain), number=REP)

            wave.to("cpu")
            fchain.to("cpu")

            for f in fchain:
                f.move_coeff("cpu")

            cpu_filter_time = timeit.timeit(lambda: cpu_filter(wave, fchain), number=REP)

            # SciPy filter coefficients
            b1, a1 = butter(2, 1000, btype="high", fs=SAMPLE_RATE)  # type: ignore
            b2, a2 = butter(2, 5000, btype="low", fs=SAMPLE_RATE)  # type: ignore
            b3, a3 = cheby1(2, 0.5, 1500, btype="high", fs=SAMPLE_RATE)  # type: ignore
            b4, a4 = cheby1(2, 0.5, 1800, btype="low", fs=SAMPLE_RATE)  # type: ignore

            scipy_filter_time = timeit.timeit(
                lambda: scipy_filter(signal, [b1, b2, b3, b4], [a1, a2, a3, a4]),
                number=REP,
            )

            print(
                f"{t},{i},{gpu_filter_time/REP:.6f},{cpu_filter_time/REP:.6f},{scipy_filter_time/REP:.6f}"
            )


if __name__ == "__main__":
    with open("iir.out", "w") as f:
        start(f)
