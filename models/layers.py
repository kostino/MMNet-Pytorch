import torch
import torch.nn as nn
from typing import List

class PreBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 32
                 ):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0))
        self.relu_a = nn.ReLU()
        self.pool_a = nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))

        self.conv_b = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
        self.relu_b = nn.ReLU()

    def forward(self, x):

        x1 = self.conv_a(x)
        x1 = self.relu_a(x1)
        x1 = self.pool_a(x1)

        x2 = self.conv_b(x)
        x2 = self.relu_b(x2)

        out = torch.cat((x1, x2), dim=1)
        return out


class MBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: List[int] = (32, 48, 48, 32),
                 pool: bool = False
                 ):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels, out_channels[0], kernel_size=(1, 1))
        self.conv_b = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=(3, 1), padding=(1, 0))
        self.relu_a = nn.ReLU()
        self.relu_b = nn.ReLU()
        self.relu_c = nn.ReLU()
        self.relu_d = nn.ReLU()

        if pool:
            self.conv_c = nn.Conv2d(out_channels[0], out_channels[2], kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
            self.conv_d = nn.Conv2d(out_channels[0], out_channels[3], kernel_size=(1, 1), stride=(1, 2))

            self.pool_b = nn.AvgPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
            self.b = nn.Sequential(self.conv_b, self.relu_b, self.pool_b)
        else:
            self.conv_c = nn.Conv2d(out_channels[0], out_channels[2], kernel_size=(1, 3), padding=(0, 1))
            self.conv_d = nn.Conv2d(out_channels[0], out_channels[3], kernel_size=(1, 1))

            self.b = nn.Sequential(self.conv_b, self.relu_b)


    def forward(self, x):

        x = self.conv_a(x)
        x = self.relu_a(x)

        x1 = self.b(x)

        x2 = self.conv_c(x)
        x2 = self.relu_c(x2)

        x3 = self.conv_d(x)
        x3 = self.relu_d(x3)

        out = torch.cat((x1, x2, x3), dim=1)
        return out

if __name__ == "__main__":
    block = MBlock(pool=False)
    inp = torch.zeros((1, 64, 128, 128))
    out = block(inp)
    print(out.size())