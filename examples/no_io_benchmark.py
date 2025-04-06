import timeit
from scipy.io import wavfile
from scipy.signal import butter, lfilter, cheby1
import torch.nn as nn
from torchfx import Wave
from torchfx.filter import HiButterworth, LoButterworth, HiShelving, LoChebyshev1


# Benchmark function for the scipy-based implementation
class SimpleWave:
    def __init__(self, ys, fs):
        self.ys = ys
        self.fs = fs

    def apply_filter(self, b, a):
        filtered = lfilter(b, a, self.ys)
        return SimpleWave(filtered, self.fs)


# Load audio data once, outside the benchmark
def load_audio_data():
    fs, data = wavfile.read("data/BERIO100.wav")
    return fs, data


fs, data = load_audio_data()

wave_gpu = Wave(data, fs)
wave_gpu.to("cuda")


# Benchmark function for the torchfx-based implementation
def benchmark_torchfx():
    fx = nn.Sequential(
        HiButterworth(1000),
        LoButterworth(2000),
        HiShelving(1200, q=0.5, gain=0.5),
        LoChebyshev1(2000),
    )
    result = wave_gpu | fx  # noqa: F841


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def shelving_highpass(cutoff, fs, gain, q):
    b, a = butter_highpass(cutoff, fs)
    return b * gain, a


def chebyshev_lowpass(cutoff, fs, order=4, ripple=0.1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = cheby1(order, ripple, normal_cutoff, btype="low", analog=False)
    return b, a


def benchmark_scipy():
    wave_cpu = SimpleWave(data, fs)
    b, a = butter_highpass(1000, wave_cpu.fs)
    wave_cpu = wave_cpu.apply_filter(b, a)
    b, a = butter_lowpass(2000, wave_cpu.fs)
    wave_cpu = wave_cpu.apply_filter(b, a)
    b, a = shelving_highpass(1200, wave_cpu.fs, gain=0.5, q=0.5)
    wave_cpu = wave_cpu.apply_filter(b, a)
    b, a = chebyshev_lowpass(2000, wave_cpu.fs)
    wave_cpu = wave_cpu.apply_filter(b, a)


# Run benchmarks
torchfx_time = timeit.timeit(benchmark_torchfx, number=10)
scipy_time = timeit.timeit(benchmark_scipy, number=10)

print(f"TorchFX implementation time: {torchfx_time:.4f} seconds")
print(f"Scipy implementation time: {scipy_time:.4f} seconds")
