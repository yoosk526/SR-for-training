import torch
import torch.nn as nn
import torch.nn.functional as F

# 입력과 출력의 크기를 같게 만드는 padding size를 정의한 합성곱 신경망 함수
def conv_layer(in_channels:int,
               out_channels:int,
               kernel_size:int,
               bias:bool=True):
    
    padding = int((kernel_size-1)/2)    # Size of Input & Output is same. (Stride = 1)
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=padding,
        bias=bias
    )

# [*, C x r^2, H, W] -> [*, C, H x r, W x r]
def pixelshuffle_blocks(in_channels,
                        out_channels,
                        upscale_factor=2,
                        kernel_size=3):
    
    # [*, C, H, W] -> [*, C x r^2, H, W], nn.PixelShuffle이 출력 채널을 r^2로 나누기 때문에 미리 늘려 놓는다.
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size)
    
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    return nn.Sequential(*[conv, pixel_shuffle])        # list 앞에 붙은 * 표시는 unpacking을 해주는 기능이다.

class ESA(nn.Module):
    # RLFB에서 esa_channels, n_feats, conv 값 받아온다.
    def __init__(self, 
                 esa_channels:int, 
                 n_feats:int, 
                 conv:nn.Module):      
        super().__init__()      # 부모 클래스(nn.Module) 호출
        
        f = esa_channels                                # esa_channels = 16
        self.conv1 = conv(n_feats, f, kernel_size=1)    # conv = nn.Conv2d
        self.conv_f = conv(f, f, kernel_size=1)         # Pointwise conv
        
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)     # [f, N, N] -> [f, (N-k+2p)/s + 1, (N-k+2p)/s + 1]
        self.conv3 = conv(f, f, kernel_size=3, padding=1)               # 크기 변하지 않음
        self.conv4 = conv(f, n_feats, kernel_size=1)                    # [f, N, N] -> [n_feats, N, N]
        
        self.sigmoid = nn.Sigmoid()     # Activation function
        # self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x:torch.Tensor):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)   # [f, N, N] -> [f, (N-k)/s + 1, (N-k)/s + 1]
        c3 = self.conv3(v_max)
        # 다시 원본 크기인 H = x.size(2), W = x.size(3)으로 Upsampling
        c3 = F.interpolate(c3 ,(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)       # [f, x.size(2), x.size(3)]
        c4 = self.conv4(c3 + cf)    # [f, x.size(3), x.size(3)]
        m = self.sigmoid(c4)
        return x * m
    
class RLFB(nn.Module):
    '''
        Residual Local Feature Block(RLFB)
    '''
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 esa_channels:int=16):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
            
        # kernel_size = 3, conv_layer 함수는 input size = output size
        self.c1_r = conv_layer(in_channels, mid_channels, 3)    
        self.c2_r = conv_layer(mid_channels, mid_channels, 3)
        self.c3_r = conv_layer(mid_channels, in_channels, 3)
        
        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)
        
        self.act = nn.LeakyReLU(0.05)   # negative slope = 0.05
        
    def forward(self, x:torch.Tensor):
        out = self.c1_r(x)
        out = self.act(out)
        
        out = self.c2_r(out)
        out = self.act(out)
        
        out = self.c3_r(out)
        out = self.act(out)
        
        out = out + x
        out = self.esa(self.c5(out))
        
        return out
        
class RLFN(nn.Module):
    def __init__(self, 
                 feature:int=52,
                 upscale:int=4):
        super().__init__()

        if feature is None:
            feature = 52    # DEFAULT
        
        self.conv_1 = conv_layer(3, feature, kernel_size=3)
        
        self.block_1 = RLFB(feature)    # 다 RLFB의 in_channels 인자만 넘겨준다.
        self.block_2 = RLFB(feature)
        self.block_3 = RLFB(feature)
        self.block_4 = RLFB(feature)
        self.block_5 = RLFB(feature)
        self.block_6 = RLFB(feature)
        
        self.conv_2 = conv_layer(feature, feature, kernel_size=3)
        self.upsampler = pixelshuffle_blocks(feature, 3, upscale_factor=upscale)    # result : channels=3, (H & W) x 4
        
    def forward(self, x:torch.Tensor):
        out_feature = self.conv_1(x)
        
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)
        out_b6 = self.block_6(out_b5)
        
        out_low_resolution = self.conv_2(out_b6) + out_feature

        # output_ = self.conv_2(out_low_resolution)
        output = self.upsampler(out_low_resolution)

        return output