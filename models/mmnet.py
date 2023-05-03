import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class CumulantEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(18, 32),
            nn.LazyBatchNorm1d()
        )


class ConvolutionalFeatureExtractor(nn.Module):
    def __init__(self, freeze: bool = True):

        super().__init__()
        pretrained = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        modules = list(pretrained.children())[:-2]
        self.feat_extractor = nn.Sequential(*modules)
        if freeze:
            for param in self.feat_extractor.parameters():
                param.requires_grad = False

        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyBatchNorm1d()
        )
        self.gmp_branch = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyBatchNorm1d()
        )

    def forward(self, x):
        x = self.feat_extractor(x)
        x1 = self.gap_branch(x)
        x2 = self.gmp_branch(x)
        out = torch.cat((x1, x2), -1)
        return out


class MMNet(nn.Module):
    def __init__(self, freeze_cnn: bool = True, num_class: int = 8):
        super().__init__()
        self.conv_feat_extract = ConvolutionalFeatureExtractor(freeze=freeze_cnn)
        self.cum_feat_extract = CumulantEncoder()
        self.num_class = num_class

        self.classifier = nn.Sequential(
            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.Linear(16, self.num_class)
        )

    def forward(self, cum, img):
        cum_f = self.cum_feat_extract(cum)
        img_f = self.conv_feat_extract(img)
        x = torch.cat((cum_f, img_f), -1)
        out = self.classifier(x)
        return out


if __name__ == "__main__":
    batch_size = 16
    img_size = (3, 224, 224)
    cum_len = 18

    img = torch.randint(0, 255, (batch_size, *img_size)).float()
    cum = torch.rand((batch_size, cum_len)).float()

    model = MMNet()
    res = model(cum, img)
    assert res.size() == (batch_size, 8)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Model total params: {}'.format(pytorch_total_params))
