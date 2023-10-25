import torch
import torch.nn as nn
from models.mmnet import CumulantEncoder


class ConvBlock(nn.Sequential):
    def __init__(self, in_nc, n_c):
        super().__init__(
            nn.Conv2d(in_nc, n_c, 3, padding=1, stride=1),
            nn.BatchNorm2d(n_c),
            nn.PReLU(),
            nn.Conv2d(n_c, n_c, 3, padding=1, stride=1),
            nn.BatchNorm2d(n_c),
            nn.PReLU(),
            nn.Conv2d(n_c, n_c, 3, padding=1, stride=1),
            nn.BatchNorm2d(n_c),
            nn.PReLU(),
            nn.MaxPool2d((2, 2))
        )


class CCNNBB(nn.Sequential):
    def __init__(self):
        super().__init__(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            nn.Conv2d(128, 128, 3, padding=1, stride=3),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=3),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )


class CCNN(nn.Module):
    def __init__(self, hocs=False):
        super().__init__()
        self.bb = CCNNBB()
        self.hocs = hocs
        if hocs:
            self.hoc_encoder = CumulantEncoder()

        self.cls = nn.Sequential(
            nn.Linear(in_features=128 if not hocs else 128 + 32
                      , out_features=512),
            nn.Dropout(0.7),
            nn.Linear(512, 64),
            nn.Dropout(0.7),
            nn.Linear(64, 8),
        )

    def forward(self, x, cum=None):
        x = self.bb(x)
        if self.hocs:
            x_c = self.hoc_encoder(cum)
            x = torch.cat([x, x_c], dim=-1)
        x = self.cls(x)

        return x


if __name__ == "__main__":
    inp = torch.zeros((10, 3, 224, 224))
    cum = torch.zeros((10, 18))
    model = CCNN()
    out = model(inp, cum)
    params = sum(p.numel() for p in model.parameters())
