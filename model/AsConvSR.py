import torch
import torch.nn as nn

from .Assembled_block import AssembledBlock

class AsConvSR(nn.Module):
    def __init__(
        self,
        kernel_size:int=3,
        stride:int=1,
        padding:int=1,
        scale_factor:int=2
    ):
        super().__init__()
        self.pixelUnShuffle = nn.PixelUnshuffle(scale_factor)                                                       # (B, 3, H, W) -> (B, 12, H/2, W/2)
        self.conv1 = nn.Conv2d(3*scale_factor**2, 32, kernel_size, stride, padding)                                 # (B, 32, H/2, W/2)
        self.assemble = AssembledBlock(32, 32, kernel_size, stride, padding, E=3, device=torch.device('cuda'))      # ()
        self.conv2 = nn.Conv2d(32, 48, kernel_size, stride, padding)                                                
        self.pixelShuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pixelUnShuffle(x)
        x = self.conv1(x)
        x = self.assemble(x)
        x = self.conv2(x)
        x = self.pixelShuffle(x)
        
        residual = x        # (1, 3, H, W)
        residual = torch.cat([residual for _ in range(self.upscale_ratio**2)], dim=1)      # (B, 3 * upscale_ratio**2, H, W)
        x = torch.add(x, residual)
        x = self.pixelShuffle(x)
        
        return x
        
if __name__ == '__main__':
    from torchsummary import summary

    model = AsConvSR()
    lr_input = torch.randn(1, 3, 128, 128)
    sr_output = model(lr_input)

    print(f"sr_output = {sr_output.shape}")    # torch.Size([1, 3, 256, 256])
    summary(model, (3, 320, 180))