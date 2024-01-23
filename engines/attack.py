import os
import argparse

import torch

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter
from utils import fig_util

import torchattacks


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--attack_name', default='', type=str, help='attack name')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
    parser.add_argument('--is_save', action='store_true', help='is save')
    parser.add_argument('--save_dir', default='', type=str, help='save directory')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.is_save and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('DEVICE:', device)
    print('ATTACK NAME:', args.attack_name)
    print('MODEL PATH:', args.model_path)
    print('-' * 50)

    # ----------------------------------------
    # attack configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    data_loader = loaders.load_data(args.data_dir, args.data_name, data_type='test')

    # ----------------------------------------
    # attack
    # ---------------------------------------
    attack = None
    if args.attack_name == 'PGDL2':
        attack = torchattacks.PGDL2(model, eps=0.5, alpha=0.2, steps=30, random_start=False)
    elif args.attack_name == 'PGD':
        attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 225, steps=30, random_start=False)
    elif args.attack_name == 'APGDL2':
        attack = torchattacks.APGD(model, norm='L2', eps=0.5, steps=30)
    elif args.attack_name == 'APGD':
        attack = torchattacks.APGD(model, norm='Linf', eps=8 / 255, steps=30)
    elif args.attack_name == 'EOTPGDL2':
        attack = attacks.EOTPGDL2(model, eps=0.5, alpha=0.2, steps=30, eot_iter=10)
    elif args.attack_name == 'EOTPGD':
        attack = torchattacks.EOTPGD(model, eps=8 / 255, alpha=2 / 255, steps=30, eot_iter=10)
    elif args.attack_name == 'FGSM':
        attack = torchattacks.FGSM(model, eps=8 / 255)
    elif args.attack_name == 'BIM':
        attack = torchattacks.BIM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif args.attack_name == 'MIFGSM':
        attack = torchattacks.MIFGSM(model, eps=8 / 255, steps=10, decay=1.0)
    elif args.attack_name == 'OnePixel':
        attack = torchattacks.OnePixel(model, pixels=1, steps=10, popsize=10, inf_batch=128)
    elif args.attack_name == 'PIM':
        attack = torchattacks.PIFGSM(model, max_epsilon=16 / 255, num_iter_set=10)
    elif args.attack_name == 'PIM++':
        attack = torchattacks.PIFGSMPP(model, max_epsilon=16 / 255, num_iter_set=10)
    elif args.attack_name == 'DIFGSM':
        attack = torchattacks.DIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif args.attack_name == 'TIFGSM':
        attack = torchattacks.TIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif args.attack_name == 'VMIFGSM':
        attack = torchattacks.VMIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=10)

    if attack is not None:
        args.mean = [0.49139968, 0.48215827, 0.44653124]
        args.std = [0.24703233, 0.24348505, 0.26158768]
        attack.set_normalization_used(mean=args.mean, std=args.std)

    acc1, class_acc = inference(data_loader, model, attack, device, args)

    print('-' * 50)
    # print(class_acc)
    print('AVG:', acc1.avg)
    print('COMPLETE !!!')


def inference(data_loader, model, attack, device, args):
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(total=len(data_loader), step=20, prefix='Test',
                             meters=[acc1_meter])
    class_acc = metrics.ClassAccuracy()
    model.eval()

    for i, samples in enumerate(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        adv_inputs = attack(inputs, labels)

        with torch.set_grad_enabled(False):
            outputs = model(adv_inputs)
            acc1, _ = metrics.accuracy(outputs, labels, topk=(1, 1))

            class_acc.update(outputs, labels)
            acc1_meter.update(acc1.item(), inputs.size(0))

            progress.display(i)

        if args.is_save:
            print('saving ...')
            save_images(adv_inputs, labels, names, args.mean, args.std, args.data_dir, args.save_dir)  # save adv images

    return acc1_meter, class_acc


def save_images(images, labels, names, mean, std, input_dir, save_dir):
    class_names = sorted([d.name for d in os.scandir(input_dir) if d.is_dir()])

    # convert the torch tensor to np image
    images = images.clone()
    mean = torch.tensor(mean).view(3, 1, 1).expand(3, images.size(2), images.size(3)).to(images.device)
    std = torch.tensor(std).view(3, 1, 1).expand(3, images.size(2), images.size(3)).to(images.device)
    images = images * std + mean
    images = images.mul(255).byte()
    images = images.cpu().numpy().transpose((0, 2, 3, 1))

    for image, label, name in zip(images, labels, names):
        class_name = class_names[label]
        save_path = os.path.join(save_dir, class_name, name)

        fig_util.save_img_by_cv2(image, save_path)


if __name__ == '__main__':
    main()
