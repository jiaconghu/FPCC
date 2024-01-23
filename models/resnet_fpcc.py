import torch
import torch.nn as nn
from core.fpcc import MSequential, CFS, CFS2d, PRO2d, SFM


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, num_classes=10):
        super().__init__()

        # residual function
        self.residual_function = MSequential(
            # ######################### CFS beg #########################
            # CFS2d(p=0.2),
            # ######################### CFS end #########################
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            # ######################### PRO beg #########################
            # PRO2d(num_classes, out_channels),
            # ######################### PRO end #########################
            # ######################### CFS end #########################
            # CFS2d(p=0.2),
            # ######################### CFS end #########################
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            # ######################### PRO beg #########################
            # PRO2d(num_classes, out_channels * BasicBlock.expansion),
            # ######################### PRO end #########################
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # shortcut
        self.shortcut = MSequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = MSequential(
                ######################### CFS beg #########################
                CFS2d(p=0.2),
                ######################### CFS end #########################
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                ######################### PRO beg #########################
                PRO2d(num_classes, out_channels * BasicBlock.expansion),
                ######################### PRO end #########################
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        # self.random = CFS2d(p=0.2)

    def forward(self, x, y=None):
        x_rf, loss1 = self.residual_function(x, y)
        x_sc, loss2 = self.shortcut(x, y)
        x = x_rf + x_sc
        # x = self.random(x)
        x = nn.ReLU(inplace=True)(x)
        loss = loss1 + loss2

        return x, loss


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, num_classes=10):
        super().__init__()

        self.residual_function = MSequential(
            ######################### CFS beg #########################
            CFS2d(p=0.2),
            ######################### CFS end #########################
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            ######################### PRO beg #########################
            PRO2d(num_classes, out_channels * BottleNeck.expansion),
            ######################### PRO end #########################
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = MSequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = MSequential(
                ######################### CFS beg #########################
                CFS2d(p=0.2),
                ######################### CFS end #########################
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                ######################### PRO beg #########################
                PRO2d(num_classes, out_channels * BottleNeck.expansion),
                ######################### PRO end #########################
                nn.BatchNorm2d(out_channels * BottleNeck.expansion),
            )

        # self.random = Random2D(p=0.2)

    def forward(self, x, y=None):
        x_rf, loss1 = self.residual_function(x, y)
        x_sc, loss2 = self.shortcut(x, y)
        x = x_rf + x_sc
        # x = self.random(x)
        x = nn.ReLU(inplace=True)(x)

        loss = loss1 + loss2
        return x, loss


class ResNet(nn.Module):

    def __init__(self, block, num_block, in_channels=3, num_classes=10):
        super().__init__()

        self.in_channels = 64

        ######################### FNA beg #########################
        self.sfm = SFM(eps=8 / 255)
        ######################### FNA end #########################

        self.conv1 = MSequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, num_classes)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, num_classes)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, num_classes)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        ######################### FRS beg #########################
        # self.cfs = FRS(p=0.2)
        ######################### FRS end #########################

        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            self.in_channels = out_channels * block.expansion

        return MSequential(*layers)

    def forward(self, x, y=None):
        x = self.sfm(x)
        output, loss1 = self.conv1(x, y)
        output, loss2 = self.conv2_x(output, y)
        output, loss3 = self.conv3_x(output, y)
        output, loss4 = self.conv4_x(output, y)
        output, loss5 = self.conv5_x(output, y)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        # output = self.cfs(output)
        output = self.fc(output)

        if self.training:
            loss_pt = loss1 + loss2 + loss3 + loss4 + loss5
            return output, loss_pt
        else:
            return output


def resnet18(in_channels=3, num_classes=10):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def resnet34(in_channels=3, num_classes=10):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


def resnet50(in_channels=3, num_classes=10):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)


def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])


def resnet152(in_channels=3, num_classes=10):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], in_channels=in_channels, num_classes=num_classes)


if __name__ == '__main__':
    print(resnet18())
