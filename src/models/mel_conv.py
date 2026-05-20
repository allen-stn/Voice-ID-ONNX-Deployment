import torch
import torch.nn as nn
import torchaudio

class MelConv(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window, persistent=False)

        # Mel filterbank
        melscale = torchaudio.transforms.MelScale(
            n_mels=n_mels,
            sample_rate=sample_rate,
            n_stft=n_fft // 2 + 1,
            norm="slaney",
        )

        fb = melscale.fb
        if fb.shape[0] != n_mels:
            fb = fb.T

        fb = fb.unsqueeze(2)

        self.mel_conv = nn.Conv1d(
            in_channels=n_fft // 2 + 1,
            out_channels=n_mels,
            kernel_size=1,
            bias=False
        )
        self.mel_conv.weight = nn.Parameter(fb, requires_grad=False)

    def forward(self, waveform):

        frames = waveform.unfold(1, self.n_fft, self.hop_length)
        frames = frames * self.window

        # enforce correct frame size
        assert frames.shape[-1] == self.n_fft, f"Frame length wrong: {frames.shape}"

        # rFFT with explicit n and dim
        spec = torch.fft.rfft(frames, n=self.n_fft, dim=-1)
        mag = spec.abs()

        mel = self.mel_conv(mag.transpose(1, 2))
        mel = torch.log(mel + 1e-6)

        return mel.transpose(1, 2)
