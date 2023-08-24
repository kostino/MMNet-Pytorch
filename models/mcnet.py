import torch
import torch.nn as nn
from typing import List
from models.layers import MBlock, PreBlock

class MCNetBackbone(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 pre_channels: int = 64,
                 block_channels: List[int] = (32, 48, 48, 32),
                 last_channels: List[int] = (32, 96, 96, 32)):
        super().__init__()

        self.inp = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=pre_channels, kernel_size=(3,7), padding=(1,3), stride=(1,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
        )

        self.pre_block = PreBlock(in_channels=pre_channels, out_channels=pre_channels//2)

        channel_sum = sum(block_channels[1:])

        self.jumpA = nn.Sequential(
            nn.Conv2d(in_channels=pre_channels, out_channels=channel_sum, kernel_size=(1, 1), stride=(1, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
        )
        self.post_pool = nn.MaxPool2d(kernel_size=(1, 3), padding=(0, 1), stride=(1, 2))
        self.MBp_A = MBlock(in_channels=pre_channels, out_channels=block_channels, pool=True)
        self.MB_A = MBlock(in_channels=channel_sum, out_channels=block_channels)

        self.jumpB = nn.Sequential(
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
        )
        self.MBp_B = MBlock(in_channels=channel_sum, out_channels=block_channels, pool=True)
        self.MB_B = MBlock(in_channels=channel_sum, out_channels=block_channels)

        self.jumpC = nn.Sequential(
            nn.ZeroPad2d((1, 0, 0, 0)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 2))
        )
        self.MBp_C = MBlock(in_channels=channel_sum, out_channels=block_channels, pool=True)
        self.MB_C = MBlock(in_channels=channel_sum, out_channels=last_channels)


    def forward(self, x):

        x = self.inp(x)
        x = self.pre_block(x)

        x1 = self.jumpA(x)
        x = self.post_pool(x)
        x = self.MBp_A(x) + x1
        x = self.MB_A(x) + x

        x = self.jumpB(x) + self.MBp_B(x)
        x = self.MB_B(x) + x

        x = self.jumpC(x) + self.MBp_C(x)
        x = torch.cat((self.MB_C(x), x), dim=1)

        return x


MCNetCFGDefault = {
    'in_channels': 1,
    'pre_channels': 64,
    'block_channels': [32, 48, 48, 32],
    'last_channels': [32, 96, 96, 32],
    'classes': 24
}

class MCNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = MCNetBackbone(
            in_channels=cfg['in_channels'],
            pre_channels=cfg['pre_channels'],
            block_channels=cfg['block_channels'],
            last_channels=cfg['last_channels']
        )

        n_feats = sum(cfg['block_channels'][1:]) + sum(cfg['last_channels'][1:])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1, -1),
            nn.Linear(in_features=n_feats, out_features=cfg['classes']),
            nn.Dropout(0.5),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class MCNetHOC(nn.Module):
    def __init__(self, cfg, hoc_channels: int = 32):
        super().__init__()

        self.backbone = MCNetBackbone(
            in_channels=cfg['in_channels'],
            pre_channels=cfg['pre_channels'],
            block_channels=cfg['block_channels'],
            last_channels=cfg['last_channels']
        )

        n_feats = sum(cfg['block_channels'][1:]) + sum(cfg['last_channels'][1:])

        self.hoc_encoder = nn.Linear(in_features=18, out_features=hoc_channels)

        self.featex = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1, -1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_feats + hoc_channels, out_features=cfg['classes']),
            nn.Dropout(0.5),
            nn.Softmax(-1)
        )

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.hoc_encoder(x)
        x1 = self.featex(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = MCNet(MCNetCFGDefault)
    inp = torch.zeros((1, 1, 2, 1024))
    out = model(inp)
    print(out.size())