import os
import argparse
import time
from copy import deepcopy

from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--data_train_dir', default='', type=str, help='data dir')
    parser.add_argument('--data_test_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    parser.add_argument('--log_dir', default='', type=str, help='log dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # if os.path.exists(args.log_dir):
    #     shutil.rmtree(args.log_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL DIR:', args.save_dir)
    # print('LOG DIR:', args.log_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load(args.model_path), strict=False)
    # model = torch.load(args.model_path)
    model.to(device)

    # modules = models.load_modules(model=model)

    train_loader = loaders.load_data(args.data_train_dir, args.data_name, data_type='train')
    test_loader = loaders.load_data(args.data_test_dir, args.data_name, data_type='test')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)

    writer = SummaryWriter(args.log_dir)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs), colour='red'):
        acc1, loss0, loss1 = train(train_loader, model, criterion, optimizer, device, args)
        writer.add_scalar(tag='training acc1', scalar_value=acc1.avg, global_step=epoch)
        writer.add_scalar(tag='training loss ce', scalar_value=loss0.avg, global_step=epoch)
        writer.add_scalar(tag='training loss pt', scalar_value=loss1.avg, global_step=epoch)
        acc1, loss0 = test(test_loader, model, criterion, device)
        writer.add_scalar(tag='test acc1', scalar_value=acc1.avg, global_step=epoch)
        writer.add_scalar(tag='test loss ce', scalar_value=loss0.avg, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc1.avg:
            best_acc = acc1.avg
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_ori.pth'))

        scheduler.step()

    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)
    print(args.save_dir)


def train(train_loader, model, criterion, optimizer, device, args):
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    loss0_meter = AverageMeter('Loss 0', ':.4e')
    loss1_meter = AverageMeter('Loss 1', ':.4e')
    progress = ProgressMeter(total=len(train_loader), step=100, prefix='Training',
                             meters=[loss0_meter, loss1_meter, acc1_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs, loss_pt = model(inputs, labels)
        loss_ce = criterion(outputs, labels)
        acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

        acc1_meter.update(acc1.item(), inputs.size(0))
        loss0_meter.update(loss_ce.item(), inputs.size(0))  # c
        loss1_meter.update(loss_pt.item(), inputs.size(0))  # 1 c

        loss_ce = loss_ce + loss_pt  # *10

        optimizer.zero_grad()  # 1
        loss_ce.backward()  # 2
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()  # 3

        progress.display(i)

    return acc1_meter, loss0_meter, loss1_meter


def test(test_loader, model, criterion, device):
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    loss0_meter = AverageMeter('Loss 0', ':.4e')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss0_meter, acc1_meter])
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss_ce = criterion(outputs, labels)
            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

            acc1_meter.update(acc1.item(), inputs.size(0))
            loss0_meter.update(loss_ce.item(), inputs.size(0))

            progress.display(i)

    return acc1_meter, loss0_meter


if __name__ == '__main__':
    main()
