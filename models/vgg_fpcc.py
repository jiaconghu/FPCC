import torch
import torch.nn as nn

from core.fpcc import MSequential, SFM, CFS, PRO, CFS2d, PRO2d

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = MSequential(
            nn.Linear(512, 4096),
            ######################### PRO beg #########################
            PRO(num_classes, 4096),
            ######################### PRO end #########################
            ######################### CFS beg #########################
            CFS(p=0.2),
            ######################### CFS end #########################
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            ######################### PRO beg #########################
            PRO(num_classes, 4096),
            ######################### PRO end #########################
            ######################### CFS beg #########################
            CFS(p=0.2),
            ######################### CFS end #########################
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x, y=None):
        x, loss1 = self.features(x, y)

        x = self.pool(x)
        x = x.view(x.size()[0], -1)

        x, loss2 = self.classifier(x, y)

        if self.training:
            loss_pt = loss1 + loss2
            return x, loss_pt
        else:
            return x


def make_layers(cfg, batch_norm=False, in_channels=3, num_classes=10):
    ######################### SFM beg #########################
    layers = [SFM(eps=8 / 255)]
    ######################### SFM end #########################

    # layers = []

    input_channel = in_channels
    for i, l in enumerate(cfg):
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        ######################### PRO beg #########################
        if i >= 1:
            layers += [PRO2d(num_classes, l)]
        ######################### PRO end #########################
        ######################### CFS beg #########################
        layers += [CFS2d(p=0.2)]
        ######################### CFS end #########################

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return MSequential(*layers)


def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16_bn(in_channels=3, num_classes=10):
    return VGG(make_layers(cfg['D'], batch_norm=True, in_channels=in_channels, num_classes=num_classes),
               num_classes=num_classes)


def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


if __name__ == '__main__':
    model = vgg16_bn(3, 10)
    print(model)
