<<<<<<< HEAD
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
        feature:int=49,
        rep:int=4,
        upscale_ratio:int=4
    ) -> None:
        super().__init__()

        if feature is None:
            feature = 49    # DEFAULT

        assert rep > 2      # rep가 2보다 작으면 AssertionError 발생, 뒤에 콤마을 적은뒤 메시지를 입력 가능
        self.upscale_ratio = upscale_ratio
        self.stem = basicConv(
            3, feature, 
        )
        buffer = []         # 반복적인 합성곱층을 간단히 작성하기 위한 변수
        for _ in range(rep-1):      # 언더스코어(_) : 인덱스 값이 굳이 필요하지 않을때 사용
            buffer.append(
                basicConv(
                    feature, feature
                )
            )
        buffer.append(basicConv(feature, 3*upscale_ratio**2, act=True))
        buffer.append(basicConv(3*upscale_ratio**2, 3*upscale_ratio**2, act=False))
        # buffer.append(nn.Conv2d(3*upscale_ratio**2, 3*upscale_ratio**2, 3, 1, 1))
        self.body = nn.Sequential(*buffer)      # "*" : Unpacking
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_ratio)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x        # (1, 3, H, W)
        residual = torch.cat([residual for _ in range(self.upscale_ratio**2)], dim=1)      # (1, 3 * upscale_ratio**2, H, W)
        x = self.stem(x)
        x = self.body(x)
        x += residual
        x = self.pixel_shuffle(x)
        x.clamp_(0, 1)

        return x


if __name__ == '__main__':
    # random_input = torch.randn(1, 3, 32, 32)
    # basic_conv = basicConv(3, 6)
    # random_output = basic_conv(random_input)
    # print(random_output.shape)
    
    net = ABPN().train()
    lr_input = torch.randn(1, 3, 320, 320)
    sr_output = net(lr_input)
    # print(net)
    print(f"sr_output = {sr_output.shape}")
=======
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
        mid_channels:int=12,
        rep:int=4,
        upscale:int=4,
        normalization=False
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = 12    # DEFAULT
        # Raise AssertionError
        assert rep > 2, "Repeat value should be larger than 2"

        self.upscale = upscale

        if not normalization:
            self.normalization = False
        else:
            self.normalization = True

        self.shallow = basicConv(3, mid_channels)
        buffer = []         # 반복적인 합성곱층을 간단히 작성하기 위한 변수
        for _ in range(rep-1):      # 언더스코어(_) : 인덱스 값이 굳이 필요하지 않을때 사용
            buffer.append(
                basicConv(mid_channels, mid_channels)
            )
        buffer.append(basicConv(mid_channels, 3*upscale**2, act=True))
        buffer.append(basicConv(3*upscale**2, 3*upscale**2, act=False))
        # buffer.append(nn.Conv2d(3*upscale_ratio**2, 3*upscale_ratio**2, 3, 1, 1))
        self.deep = nn.Sequential(*buffer)      # "*" : Unpacking
        
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x        # (1, 3, H, W)
        residual = torch.cat([residual for _ in range(self.upscale**2)], dim=1)      # (1, 3 * upscale_ratio**2, H, W)
        x = self.shallow(x)
        x = self.deep(x)
        x += residual
        x = self.pixel_shuffle(x)

        if not self.normalization:
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
>>>>>>> fec6cb4ce53e5318518c6848a7c0720eb20a1cf0
