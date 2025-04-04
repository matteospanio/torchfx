from torch import nn
import torchaudio.transforms as T
from torchfx.wave import Wave
from torchfx.filter import HiButterworth, LoButterworth


class MultiChannelEffect(nn.Module):
    ch: list[nn.Module]

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.ch = []
        self.ch.append(nn.Sequential(
            HiButterworth(1000),
            LoButterworth(2000),
        ))
        self.ch.append(nn.Sequential(
            HiButterworth(2000),
            LoButterworth(4000),
            T.Vol(0.5),
        ))

    def forward(self, x: Wave) -> Wave:
        signal = x.ys
        fs = x.fs
        for ch in range(self.num_channels):
            tmp = Wave(signal[ch], fs)
            tmp = tmp | self.ch[ch]
        return x


if __name__ == "__main__":
    wave = Wave.from_file("path/to/file.wav")
    wave.to("cuda")

    fx = MultiChannelEffect(num_channels=2)
    result = wave | fx
