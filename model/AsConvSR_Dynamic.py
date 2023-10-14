import torch
import torch.nn as nn

from .dynamic_conv import DynamicConv

class AsConvSR(nn.Module):
    def __init__(
        self,
        scale_factor:int=2,
        device=torch.device('cpu')
    ):
        super().__init__()
        self.pixelUnShuffle = nn.PixelUnshuffle(scale_factor)
        self.conv1 = nn.Conv2d(3*scale_factor**2, 32, kernel_size=3, stride=1, padding=1)
        self.assemble = DynamicConv(32, 32, E=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1)                                                
        self.pixelShuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixelUnShuffle(x)      # (B, 3, H, W) -> (B, 12, H/2, W/2)
        x = self.conv1(x)               # (B, 32, H/2, W/2)
        x = self.assemble(x)            # (B, 32, H/2, W/2)
        x = self.conv2(x)               # (B, 48, H/2, W/2)
        x = self.pixelShuffle(x)        # (B, 12, H, W)
        
        residual = x                    # (B, 3, H, W)
        residual = torch.cat([residual for _ in range(self.upscale_ratio**2)], dim=1)      # (B, 3 * upscale_ratio**2, H, W)
        x = torch.add(x, residual)
        x = self.pixelShuffle(x)        # (B, 3, H, W)
        
        return x
        
if __name__ == '__main__':
    from torchsummary import summary

    model = AsConvSR()
    lr_input = torch.randn(1, 3, 320, 180)
    sr_output = model(lr_input)

    print(f"sr_output = {sr_output.shape}")    # torch.Size([1, 3, 256, 256])
    summary(model, (3, 320, 180))