import timeit


def benchmark_torchfx():
    # Place the torchfx-based implementation here
    # Ensure that this function includes everything needed to run the benchmark
    import torchaudio
    import torch.nn as nn
    from torchfx import Wave
    from torchfx.filter import HiButterworth, LoButterworth, HiShelving, LoChebyshev1

    wave = Wave.from_file("data/BERIO100.wav")
    fx = nn.Sequential(
        HiButterworth(1000),
        LoButterworth(2000),
        HiShelving(1200, q=0.5, gain=0.5),
        LoChebyshev1(2000),
    )
    result = wave | fx
    torchaudio.save("data/BERIO100_out2.wav", result.ys, wave.fs)


def benchmark_scipy():
    # Place the scipy-based implementation here
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import butter, lfilter, cheby1

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
        # Implement a basic high-shelving filter
        # Note: Implementing a shelving filter in scipy directly might require additional steps
        # This is a placeholder for the actual filter implementation
        b, a = butter_highpass(cutoff, fs)
        return b * gain, a

    def chebyshev_lowpass(cutoff, fs, order=4, ripple=0.1):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = cheby1(order, ripple, normal_cutoff, btype="low", analog=False)
        return b, a

    class SimpleWave:
        def __init__(self, ys, fs):
            self.ys = ys
            self.fs = fs

        @classmethod
        def from_file(cls, path):
            fs, data = wavfile.read(path)
            return cls(data, fs)

        def to_file(self, path):
            wavfile.write(path, self.fs, self.ys.astype(np.int16))

        def apply_filter(self, b, a):
            filtered = lfilter(b, a, self.ys)
            return SimpleWave(filtered, self.fs)

    wave = SimpleWave.from_file("data/BERIO100.wav")
    b, a = butter_highpass(1000, wave.fs)
    wave = wave.apply_filter(b, a)
    b, a = butter_lowpass(2000, wave.fs)
    wave = wave.apply_filter(b, a)
    b, a = shelving_highpass(1200, wave.fs, gain=0.5, q=0.5)
    wave = wave.apply_filter(b, a)
    b, a = chebyshev_lowpass(2000, wave.fs)
    wave = wave.apply_filter(b, a)
    wave.to_file("data/BERIO100_out2.wav")


# Run benchmarks
torchfx_time = timeit.timeit(benchmark_torchfx, number=10)
scipy_time = timeit.timeit(benchmark_scipy, number=10)

print(f"TorchFX implementation time: {torchfx_time:.4f} seconds")
print(f"Scipy implementation time: {scipy_time:.4f} seconds")
