from ECAPA_mobile_wrapper import ECAPA_MobileWrapper
import torch
from mel_conv import MelConv
# import torch

m = MelConv()
y = m(torch.randn(1, 16000))
print(y.shape)


model = ECAPA_MobileWrapper(
    r"D:\Thesis_implement_1\Models\dense_model_stan\embedding_model.ckpt"
)
model.eval()

dummy = torch.randn(1, 16000)

torch.onnx.export(
    model,
    dummy,
    "ecapa_mobile.onnx",
    opset_version=17,
    input_names=["audio"],
    output_names=["embedding"],
    dynamic_axes={"audio": {1: "time"}}
)

print("Exported successfully!")
