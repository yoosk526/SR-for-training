import torch
import torch.nn as nn

from .dynamic_conv import DynamicConv

class AsConvDy(nn.Module):
    def __init__(
        self,
        scale_factor:int=2,
        device=torch.device('cpu')
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.pixelUnShuffle = nn.PixelUnshuffle(scale_factor)
        self.conv1 = nn.Conv2d(3*scale_factor**2, 32, kernel_size=3, stride=1, padding=1)
        self.assemble = DynamicConv(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)                                                
        self.pixelShuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                    # (B, 3, H, W)
        residual = torch.cat([residual for _ in range(self.scale_factor**2)], dim=1)      # (B, 3 * scale_factor**2, H, W)

        out = self.pixelUnShuffle(x)      # (B, 3, H, W) -> (B, 12, H/2, W/2)
        out = self.conv1(out)               # (B, 32, H/2, W/2)
        out = self.assemble(out)            # (B, 32, H/2, W/2)
        out = self.conv2(out)               # (B, 48, H/2, W/2)
        out = self.pixelShuffle(out)        # (B, 12, H, W)

        out = torch.add(out, residual)
        out = self.pixelShuffle(out)        # (B, 3, H, W)
        
        return out
        
if __name__ == '__main__':
    #from torchsummary import summary

    model = AsConvDy()
    lr_input = torch.randn(1, 3, 320, 180)
    sr_output = model(lr_input)

    print(f"sr_output = {sr_output.shape}")    # torch.Size([1, 3, 256, 256])
    #summary(model, (3, 320, 180))