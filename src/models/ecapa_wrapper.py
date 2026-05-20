import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from mel_conv import MelConv

class ECAPA_MobileWrapper(nn.Module):
    def __init__(self, model_ckpt_path):
        super().__init__()


        # --------------------------------------------------------
        # 1. ONNX-friendly mel spectrogram (no STFT!)
        # --------------------------------------------------------
        self.melspec = MelConv()   # ONNX-safe
        # print("mel shape:", mel.shape)

        # --------------------------------------------------------
        # 2. ECAPA-TDNN embedding model
        # --------------------------------------------------------
        self.ecapa = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            groups=[1, 1, 1, 1, 1],
            attention_channels=128,
            lin_neurons=192
        )

        # load weights
        state = torch.load(model_ckpt_path, map_location="cpu")
        self.ecapa.load_state_dict(state)
        self.ecapa.eval()
        # Replace SpeechBrain ASP with ONNX-safe ASP
        from onnx_pool import ONNXAttentiveStatsPool

        self.ecapa.attentive_stats_pool = ONNXAttentiveStatsPool(channels=3072,attention_channels=128)


    def forward(self, waveform):
        mel = self.melspec(waveform)  # (B, frames, 80)
        print("mel shape:", mel.shape)

        feats = torch.log(mel + 1e-6)  # still (B, frames, 80)

    # IMPORTANT: do NOT transpose here
    # ECAPA expects (B, T, 80)

        emb = self.ecapa(feats)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb
