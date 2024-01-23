import torch
from torch import nn


class MSequential(nn.Sequential):
    def forward(self, x, y=None):
        total_loss = torch.zeros(1).to(x.device)
        for module in self._modules.values():
            if module.forward.__code__.co_argcount > 2:
                x, loss = module(x, y)
                total_loss += loss
            else:
                x = module(x)
        return x, total_loss


class SFM(nn.Module):
    def __init__(self, num_channels=3, eps=8 / 255):
        super().__init__()
        mean = [0.49139968, 0.48215827, 0.44653124]
        std = [0.24703233, 0.24348505, 0.26158768]
        self.mean = torch.tensor(mean).reshape(1, num_channels, 1, 1)
        self.std = torch.tensor(std).reshape(1, num_channels, 1, 1)
        self.eps = eps

    def normalize(self, inputs):
        mean = self.mean.to(inputs.device)
        std = self.std.to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.mean.to(inputs.device)
        std = self.std.to(inputs.device)
        return inputs * std + mean

    def forward(self, x):
        if self.training:
            x_n = self.inverse_normalize(x)  # 0-1

            # -----------------------------------------------
            # x_n = x_n + torch.empty_like(x_n).uniform_(-self.eps, self.eps).to(x.device)
            # x_n = torch.clamp(x_n, min=0, max=1)
            # -----------------------------------------------
            x_n = x_n + (self.eps * torch.randn(x_n.size()).sign()).to(x.device)
            x_n = torch.clamp(x_n, min=0, max=1)
            # -----------------------------------------------

            x_n = self.normalize(x_n)
            return x_n
        else:
            return x


# fully connected layer
class CFS(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # ------------------------------------------
            mask = torch.bernoulli(torch.rand(x.size(0), x.size(1)), p=(1 - self.p)).to(x.device)  # p of 1
            x = x * mask
            # ------------------------------------------
        return x


# convolutional layer
class CFS2d(nn.Module):
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # ------------------------------------------
            mask = torch.bernoulli(torch.rand(x.size(0), x.size(1)), p=(1 - self.p)).to(x.device)  # p of 1
            x = x * mask.unsqueeze(2).unsqueeze(3)
            # ------------------------------------------
        return x


# fully connected layer
class PRO(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.patterns = nn.Parameter(torch.randn(num_classes, num_features))  # (c, d)

    def forward(self, x, y=None):
        loss = torch.tensor(0.0)
        if self.training:
            x_a = x.clone()  # (b, d)

            # ------------------------------------------
            x_mean = torch.mean(x_a, dim=1, keepdim=True)  # (b, 1)
            x_std = torch.std(x_a, dim=1, keepdim=True)  # (b, 1)
            x_p = (x_a - x_mean) / (x_std + 1e-5)  # (b, d)
            # ------------------------------------------
            # x_min, _ = torch.min(x_a, dim=1, keepdim=True)  # (b, 1)
            # x_max, _ = torch.max(x_a, dim=1, keepdim=True)  # (b, 1)
            # x_p = (x_a - x_min) / (x_max - x_min + 1e-5)  # (b, d)
            # ------------------------------------------

            p = self.patterns[y]  # (b, d)
            loss = torch.nn.L1Loss()(x_p, p)
        return x, loss


# convolutional layer
class PRO2d(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.patterns = nn.Parameter(torch.randn(num_classes, num_features))  # (c, d)

    def forward(self, x, y=None):
        loss = torch.tensor(0.0)
        if self.training:
            x_a = x.clone()  # (b, d, h, w)
            x_a = torch.mean(x_a, dim=(2, 3))  # (b, d)

            # ------------------------------------------
            x_mean = torch.mean(x_a, dim=1, keepdim=True)  # (b, 1)
            x_std = torch.std(x_a, dim=1, keepdim=True)  # (b, 1)
            x_p = (x_a - x_mean) / (x_std + 1e-5)  # (b, d)
            # ------------------------------------------
            # x_min, _ = torch.min(x_a, dim=1, keepdim=True)  # (b, 1)
            # x_max, _ = torch.max(x_a, dim=1, keepdim=True)  # (b, 1)
            # x_p = (x_a - x_min) / (x_max - x_min + 1e-5)  # (b, d)
            # ------------------------------------------

            p = self.patterns[y]  # (b, d)
            loss = torch.nn.L1Loss()(x_p, p)
        return x, loss
