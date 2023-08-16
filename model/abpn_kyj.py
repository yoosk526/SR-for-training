import torch    #Modified by kyj
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels:int, out_channels:int, 
               kernel_size:int, bias:bool=True):
    padding = int((kernel_size-1)/2)    # 입력 크기와 출력 크기가 같음(stride가 1인 경우)
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size, padding=padding, bias=bias)

class ABPN(nn.Module):
    def __init__(self, feature:int=28, upscale=3, out_channels=3):
        super().__init__()
        # Shallow feature extraction
        self.block1 = nn.Sequential(
            conv_layer(3, feature, kernel_size=3),
            nn.ReLU()
        )
        # Deep feature extraction
        self.block2 = nn.Sequential(
            conv_layer(feature, feature, kernel_size=3),
            nn.ReLU(),
            conv_layer(feature, feature, kernel_size=3),
            nn.ReLU(),
            conv_layer(feature, feature, kernel_size=3),
            nn.ReLU(),
            conv_layer(feature, feature, kernel_size=3),
            nn.ReLU(),
            conv_layer(feature, out_channels*(upscale**2), kernel_size=3),
            nn.ReLU()
        )
        # Transition
        self.block3 = conv_layer(out_channels*(upscale**2), out_channels*(upscale**2), kernel_size=3)
        # Pixel-Shuffle(depth_to_space)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x:torch.Tensor, upscale=3):
        inp = torch.permute(x, (0,2,3,1))                           # [16,3,64,64] -> [16,64,64,3]
        upsampled_inp = torch.cat([inp]*(upscale**2), dim=3)        # [16,64,64,27]
        upsampled_inp = torch.permute(upsampled_inp, (0,3,1,2))     # [16,27,64,64]

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = upsampled_inp + out               # [16,27,64,64]
        
        out = self.upsampler(out, upscale)      # [16,3,192,192]
        out = torch.clip(out, 0., 255.)

        return out
