import torch
from torch import nn, Tensor
import torchaudio.transforms as T
import torchaudio
from torchfx import FX, Wave
from torchfx.filter import HiButterworth, LoButterworth


class MultiChannelEffect(FX):
    ch: nn.ModuleList

    def __init__(self, num_channels: int, fs: int | None = None) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.fs = fs
        self.ch = nn.ModuleList(
            [
                self.channel1(),
                self.channel2(),
            ]
        )

    def channel1(self):
        return nn.Sequential(
            HiButterworth(1000, fs=self.fs),
            LoButterworth(2000, fs=self.fs),
        )

    def channel2(self):
        return nn.Sequential(
            HiButterworth(2000, fs=self.fs),
            LoButterworth(4000, fs=self.fs),
            T.Vol(0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.fs is None:
            raise ValueError("Sampling frequency (fs) must be set.")

        x_stacked = torch.stack(
            [self.ch[i](x[i]) for i in range(self.num_channels)], dim=0
        )
        return x_stacked


if __name__ == "__main__":
    # Automatically use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the audio file
    wave = Wave.from_file("data/BERIO100.wav")
    wave.to(device)

    # Create the effect and apply it to the audio
    fx = MultiChannelEffect(num_channels=2, fs=wave.fs)
    result = wave | fx

    # Save the output
    torchaudio.save("data/BERIO100_out.wav", result.ys, wave.fs)
