import torch
import torch.nn as nn
from core.fpcc import CFS, CFS2d, PRO2d, SFM, MSequential


class WideBasic(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, num_classes=10):
        super().__init__()
        self.residual = MSequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            ######################### PRO beg #########################
            PRO2d(num_classes, out_channels),
            ######################### PRO end #########################
            ######################### CFS beg #########################
            CFS2d(p=0.2),
            ######################### CFS end #########################
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ######################### PRO beg #########################
            PRO2d(num_classes, out_channels),
            ######################### PRO end #########################
        )

        self.shortcut = MSequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = MSequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                ######################### PRO beg #########################
                PRO2d(num_classes, out_channels),
                ######################### PRO end #########################
            )

        self.cfs = CFS2d(p=0.2)

    def forward(self, x, y=None):
        residual, loss1 = self.residual(x, y)
        shortcut, loss2 = self.shortcut(x, y)
        x = residual + shortcut
        x = self.cfs(x)

        loss = loss1 + loss2
        return x, loss


class WideResNet(nn.Module):
    def __init__(self, in_channels, num_classes, block, depth=50, widen_factor=1):
        super().__init__()

        self.depth = depth
        k = widen_factor
        l = int((depth - 4) / 6)
        self.in_channels = 16
        ######################### SFM beg #########################
        self.sfm = SFM(eps=8 / 255)
        ######################### SFM end #########################
        self.init_conv = nn.Conv2d(in_channels, self.in_channels, 3, 1, padding=1)
        ######################### CFS beg #########################
        self.cfs = CFS2d(p=0.2)
        ######################### CFS end #########################
        self.conv2 = self._make_layer(block, 16 * k, l, 1, num_classes)
        self.conv3 = self._make_layer(block, 32 * k, l, 2, num_classes)
        self.conv4 = self._make_layer(block, 64 * k, l, 2, num_classes)
        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        ######################### CFS beg #########################
        # self.cfs = CFS(p=0.2)
        ######################### CFS end #########################
        self.linear = nn.Linear(64 * k, num_classes)

    def forward(self, x, y=None):
        x = self.sfm(x)
        x = self.init_conv(x)
        x = self.cfs(x)
        x, loss1 = self.conv2(x, y)
        x, loss2 = self.conv3(x, y)
        x, loss3 = self.conv4(x, y)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        # x = self.cfs(x)
        x = self.linear(x)

        if self.training:
            loss_pt = loss1 + loss2 + loss3
            return x, loss_pt
        else:
            return x

    def _make_layer(self, block, out_channels, num_blocks, stride, num_classes):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, num_classes))
            self.in_channels = out_channels

        return MSequential(*layers)


# Table 9: Best WRN performance over various datasets, single run results.
def wideresnet2810(in_channels=3, num_classes=10):
    net = WideResNet(in_channels, num_classes, WideBasic, depth=28, widen_factor=10)
    return net
