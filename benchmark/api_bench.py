import timeit
import numpy as np
from scipy.signal import cheby1, butter, lfilter
from torch import nn
from torch.nn import Sequential
from torchfx import Wave
from torchfx.filter import HiChebyshev1, LoButterworth

# Parameters
SAMPLE_RATE = 44100
DURATION = 120  # 2 minutes
NUM_CHANNELS = 8


# Generate a random multichannel signal
def create_audio(sample_rate, duration, num_channels):
    signal = np.random.randn(num_channels, int(sample_rate * duration))
    signal = signal.astype(np.float32)
    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal), axis=1, keepdims=True)
    return signal


# Implementation using classes
class FilterChain(nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.f1 = HiChebyshev1(20, sample_rate)
        self.f2 = HiChebyshev1(60, sample_rate)
        self.f3 = HiChebyshev1(65, sample_rate)
        self.f4 = LoButterworth(5000, sample_rate)
        self.f5 = LoButterworth(4900, sample_rate)
        self.f6 = LoButterworth(4850, sample_rate)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        x = self.f6(x)
        return x


def apply_filters_with_classes(signal):
    fchain = FilterChain(signal.fs)
    return fchain(signal.ys)


def apply_filters_with_sequential(signal):
    fchain = Sequential(
        HiChebyshev1(20, fs=signal.fs),
        HiChebyshev1(60, fs=signal.fs),
        HiChebyshev1(65, fs=signal.fs),
        LoButterworth(5000, fs=signal.fs),
        LoButterworth(4900, fs=signal.fs),
        LoButterworth(4850, fs=signal.fs),
    )
    return fchain(signal.ys)


def apply_filters_with_pipe(signal):
    return (
        signal
        | HiChebyshev1(20)
        | HiChebyshev1(60)
        | HiChebyshev1(65)
        | LoButterworth(5000)
        | LoButterworth(4900)
        | LoButterworth(4850)
    )


def apply_filters_with_numpy_scipy(signal):
    b1, a1 = cheby1(2, 0.5, 20, btype="high", fs=SAMPLE_RATE)
    b2, a2 = cheby1(2, 0.5, 60, btype="high", fs=SAMPLE_RATE)
    b3, a3 = cheby1(2, 0.5, 65, btype="high", fs=SAMPLE_RATE)
    b4, a4 = butter(2, 5000, btype="low", fs=SAMPLE_RATE)
    b5, a5 = butter(2, 4900, btype="low", fs=SAMPLE_RATE)
    b6, a6 = butter(2, 4850, btype="low", fs=SAMPLE_RATE)

    filtered_signal = lfilter(b1, a1, signal)
    filtered_signal = lfilter(b2, a2, filtered_signal)
    filtered_signal = lfilter(b3, a3, filtered_signal)
    filtered_signal = lfilter(b4, a4, filtered_signal)
    filtered_signal = lfilter(b5, a5, filtered_signal)
    filtered_signal = lfilter(b6, a6, filtered_signal)
    return filtered_signal


def start():
    signal_data = create_audio(SAMPLE_RATE, DURATION, NUM_CHANNELS)
    wave = Wave(signal_data, SAMPLE_RATE)

    # Benchmark each method
    class_time = timeit.timeit(lambda: apply_filters_with_classes(wave), number=50)
    seq_time = timeit.timeit(lambda: apply_filters_with_sequential(wave), number=50)
    pipe_time = timeit.timeit(lambda: apply_filters_with_pipe(wave), number=50)
    scipy_time = timeit.timeit(
        lambda: apply_filters_with_numpy_scipy(signal_data), number=50
    )

    print(f"FilterChain (Classes) Time: {class_time/50:.6f}s")
    print(f"Sequential Time: {seq_time/50:.6f}s")
    print(f"Pipe Operator Time: {pipe_time/50:.6f}s")
    print(f"SciPy + NumPy Time: {scipy_time/50:.6f}s")


if __name__ == "__main__":
    start()
