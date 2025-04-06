import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter


def butter_filter(signal, cutoff, fs, btype, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return lfilter(b, a, signal)


class MultiChannelEffect:
    def __init__(self, num_channels, fs):
        self.num_channels = num_channels
        self.fs = fs

    def channel1(self, signal):
        signal = butter_filter(signal, 1000, self.fs, btype="high")
        signal = butter_filter(signal, 2000, self.fs, btype="low")
        return signal

    def channel2(self, signal):
        signal = butter_filter(signal, 2000, self.fs, btype="high")
        signal = butter_filter(signal, 4000, self.fs, btype="low")
        signal = 0.5 * signal  # Volume attenuation
        return signal

    def apply(self, data):
        if self.fs is None:
            raise ValueError("Sampling frequency (fs) must be set.")

        output = np.copy(data)
        if self.num_channels != data.shape[0]:
            raise ValueError(
                f"Expected {self.num_channels} channels but got {data.shape[0]}."
            )

        output[0] = self.channel1(data[0])
        output[1] = self.channel2(data[1])
        return output


if __name__ == "__main__":
    import os

    input_path = "data/BERIO100.wav"
    output_path = "data/BERIO100_out.wav"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file {input_path} not found.")

    fs, data = wavfile.read(input_path)
    print(f"Loaded '{input_path}' with shape {data.shape}, fs={fs}")

    # Convert to float32 in [-1, 1] if necessary
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Ensure data is (channels, samples)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    elif data.shape[1] < data.shape[0]:
        data = data.T  # ensure shape is (channels, samples)

    fx = MultiChannelEffect(num_channels=2, fs=fs)
    result = fx.apply(data)

    # Back to int16 for WAV export
    result_out = (result * 32767).astype(np.int16).T
    wavfile.write(output_path, fs, result_out)
    print(f"Saved processed audio to '{output_path}'")
