import torch
import argparse
import numpy as np
import cv2
from model import abpn
from model import rlfn
from model import innopeak

def preprocess(x:np.ndarray):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = np.transpose(x, [2, 0, 1])      # [H, W, C] -> [C, H, W]
    # x = x.astype(np.float32) / 255.0    
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x

def postprocess(x:np.ndarray):
    # x = (x * 255.0).astype(np.uint8)
    x = x.astype(np.float32)
    x = np.transpose(x, [1, 2, 0])      # [C, H, W] -> [H, W, C]
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x

def bicubicResize(x:np.ndarray, scale:int=4):
    h, w, _ = x.shape
    x = cv2.resize(x, dsize=(h*scale, w*scale), interpolation=cv2.INTER_NEAREST)
    return x

def horizontalFusion(bi:np.ndarray, sr:np.ndarray):
    assert bi.shape == sr.shape, "Check image shape"
    h, w, c = bi.shape
    canvas = np.zeros_like(bi).astype(np.uint8)
    canvas[:, 0:w//2, :] = bi[:, 0+480:w//2+480, :]
    canvas[:, w//2:w, :] = sr[:, 0+480:w//2+480, :]
    return canvas

def openImage(filepath):
    try:
        imgObj = cv2.imread(filepath, cv2.IMREAD_COLOR)
        imgObj = cv2.cvtColor(imgObj, cv2.COLOR_BGR2RGB)
        return imgObj
    except:
        raise ValueError()

def npToTensor(x:np.ndarray):
    x = np.transpose(x, [2, 0, 1])
    return torch.from_numpy(x)

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--weight", type=str, default="./run/abpn-231002-21-38/weights/abpn_final.pth"
)
parser.add_argument(
	"--model", type=str, choices=['rlfn', 'abpn', 'innopeak'], default='abpn'
)
parser.add_argument(
	"--scale", type=int, default=4
)

if __name__ == "__main__":
    args = parser.parse_args()

    model = args.model
    if model == 'rlfn':
        model = rlfn.RLFN()
    elif model == 'abpn':
        model = abpn.ABPN()
    elif model == 'innopeak':
        model = innopeak.InnoPeak(upscale=args.scale)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        model.load_state_dict(torch.load(args.weight)).cuda()
    else:
        model.load_state_dict(torch.load(args.weight), map_location='cpu')

    model.eval()
    imgToTensor = npToTensor(openImage(args.image)).unsqueeze(0).float()
    with torch.no_grad():
        srObj = model(imgToTensor).squeeze(0).detach().numpy().astype(np.uint8)
    srObj = np.transpose(srObj, [1, 2, 0])
    srObj = cv2.cvtColor(srObj, cv2.COLOR_RGB2BGR)

    cv2.imshow("PyTorch test", srObj)
    cv2.waitKey()
    cv2.destroyAllWindows()


    

#     LR_WINDOW = "LR_WINDOW"
#     BICUBIC_SR_WINDOW = "BICUBIC vs SUPER-RESOLUTION"

#     cv2.namedWindow(LR_WINDOW)
#     cv2.namedWindow(BICUBIC_SR_WINDOW)
#     cv2.moveWindow(LR_WINDOW, 30, 20)
#     cv2.moveWindow(BICUBIC_SR_WINDOW, 800, 300)

# model.load_state_dict(torch.load(PATH))
# print(model.parameters)

# lrObj = cv2.imread('./image/lr_img_01.png')
# bicubic = bicubicResize(lrObj)
# input_np = preprocess(lrObj).numpy()
# sr_np = postprocess(model(input_np))

# canvas = horizontalFusion(bicubic, sr_np)

# cv2.imshow(BICUBIC_SR_WINDOW, canvas)
# cv2.imshow(LR_WINDOW, lrObj)

# cv2.waitKey()
# cv2.destroyAllWindows()