import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
from einops import rearrange


class ExampleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.p1 = nn.Parameter(torch.randn(1, 1, 1, 4))
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(4, 8, 3, 1, 1)
        self.bn23 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1, groups=16)
        self.bn4 = nn.BatchNorm2d(16)
        self.conv5_1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5_2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.identity = nn.Identity()
        self.bn5i = nn.BatchNorm2d(16)
        self.bn45 = nn.BatchNorm2d(32)
        self.fcp = nn.Linear(32, 32)
        self.conv6 = nn.Conv2d(32, 1, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(1)
        self.p6 = nn.Parameter(torch.randn(1, 1, 1, 1))
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.AvgPool2d(2, 2)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x + self.p1
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv2(x1)
        x2 = self.conv3(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn23(x)
        identity = self.identity(x)
        x1 = self.conv4(x)
        x1 = self.bn4(x1)
        x2 = self.conv5_1(x)
        x2 = self.conv5_2(x2 + identity)
        x2 = self.bn5i(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.bn45(x)  # (1, 32, 4, 4)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.fcp(x)  # (1, 4, 4, 32) -> (1, 4, 4, 32)
        x = rearrange(x, "b (h w) c -> b c h w", h=4)
        x = self.conv6(x)  # (1, 32, 4, 4)
        x = self.bn6(x)
        x = torch.nn.functional.relu(x)
        x = x * self.p6
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class ShuffleAttention(nn.Module):
    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w
        # DONE: if split before norm, the backward is interupted
        #       Now, for the single module test, give the input with requires_grad=True
        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


if __name__ == "__main__":
    input = torch.randn(50, 32, 224, 224)
    se = ShuffleAttention(channel=32, G=4)
    output = se(input)
    print(output.shape)
