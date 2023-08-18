import os
import torch
import torch.nn as nn
import torch.onnx
import onnx
from onnx import shape_inference
import torchvision.transforms as T
import time
from PIL import Image
from model import abpn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def Convert_ONNX(model:nn.Module, input_shape:tuple, name:str):
    if os.path.exists(name):
        os.remove(name)
        print(f"[WARN] REMOVE FILE = {name}")
    if not name.endswith('.onnx'):
        name = name + '.onnx'

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(input_shape, requires_grad=True).to(device)

    # Export the model
    torch.onnx.export(  model,                     # model being run
                        dummy_input,               # model input (or a tuple for multiple inputs)
                        name,                      # where to save the model
                        export_params=True,        # store the trained parameter weights inside the model file
                        input_names = ['input'],   # the model's input names
                        output_names = ['output']  # the model's output names
    )

    # Add shape to onnx model
    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(name)), name)

    print('# Model has been converted to ONNX')
    print(f'# CREATE NEW ONNX FILE = {name}')

weights = "./run/train/abpn-230818-20-12/weights/abpn_final.pth"

model = abpn.ABPN()

model.load_state_dict(torch.load(weights))

model.to(device)

Convert_ONNX(model, (1, 3, 224, 320), "x4_224_320.onnx")    # (B, C, H, W)