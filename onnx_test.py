import cv2
import numpy as np
import argparse
import onnxruntime as ort
from utils.data_utils import *
from model import abpn, rlfn, innopeak

parser = argparse.ArgumentParser()
parser.add_argument(
	"--image", type=str, default="./media/ori/270_480_01.png"
)
parser.add_argument(
	"--onnx", type=str, default="./onnx/x4_270_480_abpn_007.onnx"
)
parser.add_argument(
	"--scale", type=int, default=4
)
parser.add_argument(
	"--norm", action="store_true"
)

if __name__ == "__main__":
    opt = parser.parse_args()

    lrObj = preprocess(openImage(opt.image), opt.norm)
    print(lrObj)

    ort_sess = ort.InferenceSession(opt.onnx, providers=['CUDAExecutionProvider'])

    output = ort_sess.run(None, {'input': lrObj})[0]
    print(output)
    srObj = postprocess(output, opt.norm)
    
    cv2.imshow("SuperResolution", srObj)
    cv2.waitKey()
    cv2.destroyAllWindows()