import torch
import torch.nn as nn

class basicConv(nn.Module):    
    def __init__(
        self,
        in_feat:int,
        out_feat:int,
        kernel_size:int=3,
        stride:int=1,
        padding:int=1,
        act:bool=True
    ) -> None:  # return 값이 없음을 명시
        super().__init__()
        self.conv = nn.Conv2d(
            in_feat, out_feat, kernel_size, stride, padding
        )
        if act:     # Activation 함수 실행 유무
            self.act = nn.ReLU()
            # self.act = self.myReLU
        else:
            self.act = None    
    
    # def myReLU(self, x:torch.Tensor) -> torch.Tensor:
    #     return torch.maximum(x, torch.zeros_like(x))
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ABPN(nn.Module):
    def __init__(
        self,
        mid_feat:int=49,
        rep:int=4,
        upscale_ratio:int=4,
        norm=False
    ) -> None:
        super().__init__()

        if mid_feat is None:
            mid_feat = 49    # DEFAULT
        # Raise AssertionError
        assert rep > 2, "Repeat value should be larger than 2"

        self.upscale_ratio = upscale_ratio

        if not norm:
            self.norm = False
        else:
            self.norm = True

        self.shallow = basicConv(3, mid_feat)
        buffer = []         # 반복적인 합성곱층을 간단히 작성하기 위한 변수
        for _ in range(rep-1):      # 언더스코어(_) : 인덱스 값이 굳이 필요하지 않을때 사용
            buffer.append(
                basicConv(mid_feat, mid_feat)
            )
        buffer.append(basicConv(mid_feat, 3*upscale_ratio**2, act=True))
        buffer.append(basicConv(3*upscale_ratio**2, 3*upscale_ratio**2, act=False))
        # buffer.append(nn.Conv2d(3*upscale_ratio**2, 3*upscale_ratio**2, 3, 1, 1))
        self.deep = nn.Sequential(*buffer)      # "*" : Unpacking
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_ratio)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x        # (1, 3, H, W)
        residual = torch.cat([residual for _ in range(self.upscale_ratio**2)], dim=1)      # (1, 3 * upscale_ratio**2, H, W)
        x = self.shallow(x)
        x = self.deep(x)
        x += residual
        x = self.pixel_shuffle(x)

        if not self.norm:
            x = x.clamp(0.0, 255.0)
        else:
            x = x.clamp(0, 1)

        return x


if __name__ == '__main__':
    # random_input = torch.randn(1, 3, 32, 32)
    # basic_conv = basicConv(3, 6)
    # random_output = basic_conv(random_input)
    # print(random_output.shape)
    
    net = ABPN().train()
    print(net)
    lr_input = torch.randn(1, 3, 320, 320)
    sr_output = net(lr_input)
    print(f"sr_output = {sr_output.shape}")