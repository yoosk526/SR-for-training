import torch
import numpy as np
import cv2
from model import abpn

def preprocess(x:np.ndarray):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, [2, 0, 1])      # [H, W, C] -> [C, H, W]
    x = x.astype(np.float32) / 255.0    # Normalization
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x

def postprocess(x:np.ndarray):
    # Denormalization 필요 (*255)
    x = (x * 255.0).astype(np.uint8)
    x = np.transpose(x, [1, 2, 0])      # [C, H, W] -> [H, W, C]
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def bicubicResize(x:np.ndarray, scale:int=3):
    h, w, _ = x.shape
    x = cv2.resize(x, dsize=(w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    return x

def horizontalFusion(bi:np.ndarray, sr:np.ndarray):
    assert bi.shape == sr.shape
    h, w, c = bi.shape
    canvas = np.zeros_like(bi).astype(np.uint8)
    canvas[:, 0:w//2, :] = bi[:, 0+200:w//2+200, :]
    canvas[:, w//2:w, :] = sr[:, 0+200:w//2+200, :]
    return canvas

PATH = "./abpn_final.pth"

LR_WINDOW = "LR_WINDOW"
BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"

cv2.namedWindow(LR_WINDOW)
cv2.namedWindow(BICUBIC_SR_WINDOW)
cv2.moveWindow(LR_WINDOW, 30, 20)
cv2.moveWindow(BICUBIC_SR_WINDOW, 800, 300)

model = abpn.ABPN()
model.load_state_dict(torch.load(PATH))
print(model.parameters)

lrObj = cv2.imread('./image/lr_img_01.png')
bicubic = bicubicResize(lrObj)
input_np = preprocess(lrObj).numpy()
sr_np = postprocess(model(input_np))

canvas = horizontalFusion(bicubic, sr_np)

cv2.imshow(BICUBIC_SR_WINDOW, canvas)
cv2.imshow(LR_WINDOW, lrObj)

cv2.waitKey()
cv2.destroyAllWindows()