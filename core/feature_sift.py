import sys

import numpy as np

sys.path.append('.')

import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import pickle

import loaders
import models


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs


def grads(outputs, inputs, retain_graph=True, create_graph=False):
    return torch.autograd.grad(outputs=outputs,
                               inputs=inputs,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]


class FeatureSift:
    def __init__(self, modules, num_classes, num_samples, feature_type='a+', is_high_confidence=True):
        self.modules = [HookModule(module) for module in modules]  # [L, C, N, [channels]]
        self.num_samples = num_samples

        self.values = [[[] for _ in range(num_classes)] for _ in range(len(modules))]  # [L, C, [N, D]]
        self.scores = torch.zeros((len(modules), num_classes, num_samples))  # [L, C, N]
        self.nums = torch.zeros((len(modules), num_classes), dtype=torch.long)  # [L, C]

        self.feature_type = feature_type
        self.is_high_confidence = is_high_confidence

    def __call__(self, outputs, labels, layers=None):
        softmaxs = nn.Softmax(dim=1)(outputs.detach())  # (b, c)

        if layers is None:
            layers = range(len(self.modules))

        for layer in layers:
            module = self.modules[layer]
            values = module.outputs  # 'a'

            if 'g' in self.feature_type:
                nll_loss = nn.NLLLoss()(outputs, labels)
                values = grads(-nll_loss, module.outputs)

            if '+' in self.feature_type:
                values = torch.relu(values)
            elif '-' in self.feature_type:
                values = torch.relu(-values)
            elif '|' in self.feature_type:
                values = torch.abs(values)

            if isinstance(module.module, nn.Conv2d):  # [b, d, h, w]
                values = torch.mean(values, dim=(2, 3))  # [b, d]
                # values = nn.AdaptiveMaxPool2d(1)(values).squeeze(3).squeeze(2)  # [b, d]
            elif isinstance(module.module, nn.Linear):  # [b, d]
                values = values  # [b, d]

            values = values.detach().cpu().numpy()

            for i, label in enumerate(labels):  # each datas
                score = softmaxs[i][label]  # (b, c) -> ()

                if self.is_high_confidence:  # sift high confidence
                    if self.nums[layer][label] == self.num_samples:
                        score_min, index = torch.min(self.scores[layer][label], dim=0)
                        if score > score_min:
                            self.values[layer][label][index] = values[i]
                            self.scores[layer][label][index] = score
                    else:
                        self.values[layer][label].append(values[i])
                        self.scores[layer][label][self.nums[layer][label]] = score
                        self.nums[layer][label] += 1
                else:  # sift low confidence
                    if self.nums[layer][label] == self.num_samples:
                        score_max, index = torch.max(self.scores[layer][label], dim=0)
                        if score < score_max:
                            self.values[layer][label][index] = values[i]
                            self.scores[layer][label][index] = score
                    else:
                        self.values[layer][label].append(values[i])
                        self.scores[layer][label][self.nums[layer][label]] = score
                        self.nums[layer][label] += 1

    def save(self, layers, save_dir, save_name):  # [l, c, n, d]
        if layers is None:
            layers = range(len(self.modules))

        for layer in layers:
            values = self.values[layer]  # [c, n, d]
            values = np.asarray(values)
            print(values.shape)

            save_path = os.path.join(save_dir, 'layer{}_{}.pkl'.format(layer, save_name))
            values_file = open(save_path, 'wb')
            pickle.dump(values, values_file)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default=10, type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--num_samples', default=10, type=int, help='num samples')
    parser.add_argument('--feature_type', default='', type=str, help='feature type')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    parser.add_argument('--save_name', default='', type=str, help='save name')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    assert args.feature_type in ['a', 'a+', 'a-', 'a|', 'g', 'g+', 'g-', 'g|']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('SAVE PATH:', args.save_dir)
    print('SAVE NAME:', args.save_name)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(data_dir=args.data_dir, data_name=args.data_name, data_type='test')

    modules = models.load_modules(model=model)

    feature_sift = FeatureSift(modules=modules,
                               num_classes=args.num_classes,
                               num_samples=args.num_samples,
                               feature_type=args.feature_type,
                               is_high_confidence=True)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for samples in tqdm(data_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        if 'a' in args.feature_type:
            outputs = model(inputs)
        else:
            with torch.no_grad():
                outputs = model(inputs)

        feature_sift(outputs=outputs, labels=labels, layers=None)

    feature_sift.save(layers=None, save_dir=args.save_dir, save_name=args.save_name)


if __name__ == '__main__':
    main()
