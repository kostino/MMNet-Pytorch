import torch
import torch.nn as nn
from typing import List
from models.mmnet import CumulantEncoder
from common.train_utils import _init_weights


class FiFBLock(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 hid_channels: int = 32):
        super().__init__()

        self.conv_a = nn.Conv2d(in_channels, hid_channels, kernel_size=(1, 1), padding='same')
        self.conv_b = nn.Conv2d(in_channels, hid_channels, kernel_size=(1, 1), padding='same')
        self.conv_a_act = nn.ReLU()
        self.conv_b_act = nn.ReLU()

        self.gc_31 = nn.Conv2d(hid_channels, hid_channels, kernel_size=(3, 1), padding='same', groups=hid_channels)
        self.gc_13 = nn.Conv2d(hid_channels, hid_channels, kernel_size=(1, 3), padding='same', groups=hid_channels)
        self.gc_31_act = nn.Hardtanh(0, 3)
        self.gc_13_act = nn.Hardtanh(0, 3)

        self.c_31 = nn.Conv2d(hid_channels, hid_channels, kernel_size=(3, 1), padding='same')
        self.c_13 = nn.Conv2d(hid_channels, hid_channels, kernel_size=(1, 3), padding='same')
        self.c_31_act = nn.ReLU()
        self.c_13_act = nn.ReLU()

        self.c_a_end = nn.Conv2d(2 * hid_channels, hid_channels, kernel_size=(1, 1), padding='same')
        self.c_b_end = nn.Conv2d(2 * hid_channels, hid_channels, kernel_size=(1, 1), padding='same')
        self.c_a_end_act = nn.ReLU()
        self.c_b_end_act = nn.ReLU()

    def forward(self, x):
        x1 = self.conv_a_act(self.conv_a(x))
        x2 = self.conv_b_act(self.conv_b(x))

        x11 = self.gc_13_act(self.gc_13(x1))
        x12 = self.gc_31_act(self.gc_31(x1))
        x1 = torch.cat((x11, x12), dim=1)

        x21 = self.c_13_act(self.c_13(x2))
        x22 = self.c_31_act(self.c_31(x2))
        x2 = torch.cat((x21, x22), dim=1)

        x1 = self.c_a_end_act(self.c_a_end(x1))
        x2 = self.c_b_end_act(self.c_b_end(x2))

        x = torch.cat((x1, x2), dim=1)

        return x


class FiFModule(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 hid_channels: int = 64):
        super().__init__()

        self.fif_a = FiFBLock(hid_channels, hid_channels // 2)
        self.fif_b = FiFBLock(hid_channels, hid_channels // 2)
        self.gc = nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 3), groups=hid_channels, stride=(2, 2), padding=(1, 1))
        self.gc_act = nn.Hardtanh(0, 3)

    def forward(self, x):
        x = self.gc_act(self.gc(x))
        x = self.fif_a(x) + x
        x = self.fif_b(x)

        return x

class FiFNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 1, hid_channels: int = 64, n_modules: int = 5):
        super().__init__()
        self.c = nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 3), padding='same')
        self.mods = nn.ModuleList([
            FiFModule(hid_channels, hid_channels) for _ in range(n_modules)
        ])

    def forward(self, x):
        x = self.c(x)
        for mod in self.mods:
            x = mod(x)

        return x


class FiFNet(nn.Module):
    def __init__(self, classes, in_channels: int = 1, hid_channels: int = 64, n_modules: int = 5, hocs=False):
        super().__init__()
        self.backbone = FiFNetBackbone(in_channels, hid_channels, n_modules)
        self.hocs = hocs
        if hocs:
            self.hoc_encoder = CumulantEncoder()
        self.featex = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1, -1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hid_channels if not hocs else hid_channels + 32
                      , out_features=128),
            nn.Linear(in_features=128, out_features=classes),
            # nn.Softmax(-1)
        )
        self.apply(_init_weights)

    def forward(self, x, cum=None):
        x = self.backbone(x)
        x = self.featex(x)
        if self.hocs:
            x_c = self.hoc_encoder(cum)
            x = torch.cat([x, x_c], dim=-1)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = FiFNet(8)
    inp = torch.zeros((10, 1, 224, 224))
    out = model(inp)
    fif_params = sum(p.numel() for p in model.parameters())
    model = FiFNet(8, hocs=True)
    hocs = torch.zeros((10, 18))
    out = model(inp, hocs)
    fif_params_hoc = sum(p.numel() for p in model.parameters())
