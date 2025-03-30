from torch import nn
import torchaudio.transforms as T
from torchfx.wave import Wave
from torchfx.filter.iir import HighPass, LowPass


class MultiChannelEffect(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x: Wave) -> Wave:
        for ch in range(self.num_channels):
            x[ch] = x[ch] | self.high_pass | self.low_pass
        return x

    def ch1(self, x: Wave) -> Wave:
        x = x | HighPass(1000) | LowPass(2000)
        return x

    def ch2(self, x: Wave) -> Wave:
        x = x | HighPass(2000) | LowPass(4000) | T.Vol(0.5)
        return x


if __name__ == "__main__":
    wave = Wave.from_file("path/to/file.wav")
    wave.to("cuda")

    fx = MultiChannelEffect(num_channels=2)
    result = fx(wave)
