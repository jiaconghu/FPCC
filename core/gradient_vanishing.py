import argparse

import torch
from torch import nn

import loaders
import models


# def check(inputs, labels):
#     inputs = inputs.clone().detach().to(self.device)
#     labels = labels.clone().detach().to(self.device)

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.load_model(args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    data_loader = loaders.load_data(args.data_dir, args.data_name, data_type='test')

    grads = []
    for i, samples in enumerate(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs.requires_grad = True

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        grad = torch.autograd.grad(
            loss, inputs, retain_graph=False, create_graph=False
        )[0]

        grads.append(torch.norm(grad, p=2))

    print(torch.mean(torch.asarray(grads)))


if __name__ == '__main__':
    main()
