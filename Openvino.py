from options.base_options import BaseOptions
from model import AFWNET
import torch
import warnings
warnings.filterwarnings("ignore")

model_path = "checkpoints/quantized/tryon.pth"
opt = BaseOptions().parse()
# load model
tryon_model = AFWNET(opt)
tryon_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# random input data
x1 = torch.randn(1, 3, 256, 192)
x2 = torch.randn(1, 3, 256, 192)
x3 = torch.randn(1, 1, 256, 192)

x1 = torch.autograd.Variable(x1)
x2 = torch.autograd.Variable(x2)
x3 = torch.autograd.Variable(x3)
print("This is a sign.")
output_path = 'checkpoints/onnx/TryonModel.onnx'

torch.onnx.export(
    tryon_model,
    (x1, x2, x3),
    output_path,
    verbose=True,
    do_constant_folding=True,
    opset_version=16
)