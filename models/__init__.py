import torch
from models import vgg, vgg_fpcc, resnet, resnet_fpcc, wideresnet, wideresnet_fpcc
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 50)
    print('MODEL NAME:', model_name)
    print('NUM CLASSES:', num_classes)
    print('-' * 50)

    model = None
    if model_name == 'vgg16':
        model = vgg.vgg16_bn(in_channels, num_classes)
    if model_name == 'vgg16_fpcc':
        model = vgg_fpcc.vgg16_bn(in_channels, num_classes)
    if model_name == 'resnet50':
        model = resnet.resnet50(in_channels, num_classes)
    if model_name == 'resnet50_fpcc':
        model = resnet_fpcc.resnet50(in_channels, num_classes)
    if model_name == 'wrn2810':
        model = wideresnet.wideresnet2810(in_channels, num_classes)
    if model_name == 'wrn2810_fpcc':
        model = wideresnet_fpcc.wideresnet2810(in_channels, num_classes)
    return model


def load_modules(model, model_layers=None):
    assert model_layers is None or type(model_layers) is list

    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            modules.append(module)
        if isinstance(module, torch.nn.Linear):
            modules.append(module)

    modules.reverse()  # reverse order
    if model_layers is None:
        model_modules = modules
    else:
        model_modules = []
        for layer in model_layers:
            model_modules.append(modules[layer])

    print('-' * 50)
    print('Model Layers:', model_layers)
    print('Model Modules:', model_modules)
    print('Model Modules Length:', len(model_modules))
    print('-' * 50)

    return model_modules
