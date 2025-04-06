import torch
from torch import nn, Tensor
import torchaudio.transforms as T
import torchaudio
from torchfx.wave import Wave
from torchfx.filter import HiButterworth, LoButterworth


class MultiChannelEffect(nn.Module):
    ch: list[nn.Module]

    def __init__(self, num_channels: int, fs: int) -> None:
        super().__init__()
        print("MultiChannelEffect - init")
        self.num_channels = num_channels
        self.fs = fs
        self.ch = []
        self.ch.append(self.channel1())
        self.ch.append(self.channel2())

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
        print("MultiChannelEffect - forward")
        for i in range(self.num_channels):
            print(f"MultiChannelEffect - forward channel {i}")
            x[i] = self.ch[i](x[i])

        return x


if __name__ == "__main__":
    print("CUDA disponibile:", torch.cuda.is_available())
    print("Dispositivi CUDA:", torch.cuda.device_count())
    print(
        "Nome device:",
        (
            torch.cuda.get_device_name(0)
            if torch.cuda.device_count() > 0
            else "Nessun dispositivo"
        ),
    )

    wave = Wave.from_file("data/BERIO100.wav")
    wave.to("cuda")
    fx = MultiChannelEffect(num_channels=2, fs=wave.fs)
    result = wave | fx
    torchaudio.save("BERIO100_out.wav", result.ys, wave.fs)
