import torch
from torch import nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(
        self,
        in_planes,
        ratio,
        K,
        temperature=30,
        init_weight=True
    ):
        super().__init__()
        self.temperature = temperature

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        assert in_planes > ratio
        hidden_planes = in_planes//ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_planes, K, kernel_size=1, bias=False)
        )

        if(init_weight):
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)                       # (B, C, H, W) -> (B, C, 1, 1)
        att = self.conv(att).view(x.shape[0], -1)    # (B, K, 1, 1) -> (B, K)
        return F.softmax(att/self.temperature, -1)   # (B, K)

class DynamicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding=0,
        dilation=1,
        grounps=1,
        bias=True,
        K=4,
        temperature=30,
        ratio=4,
        init_weight=True
    ):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K

        self.attention = Attention(in_planes=in_planes, ratio=ratio, K=K, temperature=temperature, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K,out_planes,in_planes//grounps,kernel_size,kernel_size), requires_grad=True)
        if(bias):
            self.bias = nn.Parameter(torch.randn(K,out_planes), requires_grad=True)
        else:
            self.bias = None
        
        if(init_weight):
            self._initialize_weights()

        #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self,x):
        bs,in_planes,h,w = x.shape
        x = x.view(1,-1,h,w)                # (B, C, H, W)
        softmax_att = self.attention(x)     # (B, K)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)  # (B*out, in, k, k)

        if(self.bias is not None):
            bias = self.bias.view(self.K, -1)                      # (K, out)
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # (B, out)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, groups=self.groups*bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups*bs, dilation=self.dilation)
        
        output = output.view(bs,self.out_planes,h,w)                # (B, C, H, W)
        return output

if __name__ == '__main__':
    input = torch.randn(2,32,64,64)
    m = DynamicConv(in_planes=32,out_planes=64,kernel_size=3,stride=1,padding=1,bias=False)
    out = m(input)
    print(out.shape)