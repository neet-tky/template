import numpy as np
import math
import argparse

import torch
from torch import nn, optim
from torch.utils import tensorboard

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets.cifar as CIFAR
from torchvision.models import vgg

def train():
    losses, correct = 0, 0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        pred_y = model(x)
        loss = criterion(pred_y, y)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # loss, acc
        _, predicted = torch.max(pred_y, 1)
        correct += (y == predicted).sum().item()

        losses += loss.item()

    return losses / len_train, correct / len_train

def valid():
    losses, correct = 0, 0

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            pred_y = model(x)
            loss =criterion(pred_y, y)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # loss, acc
            _, predicted = torch.max(pred_y, 1)
            correct += (predicted == y).sum().item()

            losses += loss.item()
    return losses / len_test, correct / len_test

def test():
    confusion_matrix = torch.zeros(args.class_num, args.class_num)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            pred_y = model(x)
            loss = criterion(pred_y, y)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            # loss, acc
            _, predicted = torch.max(pred_y, 1)
            for t, p in zip(y.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix

def param_decay(epoch):
    if epoch < 100:
        return args.lr

    elif epoch < 150:
        return args.lr * .1

    else:
        return args.lr * .01

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=bool, default=False, help='use gpu or not')
    parser.add_argument('-lr', type=float, default=.01, help='initial learning rate')
    parser.add_argument('-batch', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-size', type=int, default=32, help='image size for datasets')
    parser.add_argument('-ch', type=int, default=3, help='input channels')
    parser.add_argument('-class_num', type=int, default=10, help='data class')
    parser.add_argument('-epoch', type=int, default=200, help='training epoch')
    parser.add_argument('-load', type=str, help='load weight')
    parser.add_argument('-save', type=str, default='save/model.pth', help='saved models')
    parser.add_argument('-fmodel', type=str, default='save/fmodel', help='final saved models')
    parser.add_argument('-lr_scheduler')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    prepro = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.RandomCrop(args.size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    val_prepro = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        #transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    trainset = CIFAR.CIFAR10(root='~/Datasets/', train=True, transform=prepro, target_transform=None, download=True)
    testset = CIFAR.CIFAR10(root='~/Datasets/', train=False, transform=val_prepro, target_transform=None, download=True)

    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    len_train = len(trainset)
    len_test = len(testset)

    model = vgg.vgg16(pretrained=False)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, args.class_num)

    if args.load is not None:
        checkpoints = torch.load(args.load)
        weights = checkpoints['weights']
        max_loss = checkpoints['loss']
        start = checkpoints['epoch']
        model.load_state_dict(weights)

    else:
        # init_weights
        max_loss = math.inf
        start = 0


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=param_decay)

    writer = tensorboard.SummaryWriter()
    for epoch in range(start, args.epoch):
        scheduler.step()

        t_loss, t_acc = train()
        v_loss, v_acc = valid()

        print(epoch, t_loss, v_loss, t_acc, v_acc)

        writer.add_scalars(
            'data/loss',
            {'train/loss': t_loss, 'valid/loss': v_loss},
            epoch + 1
        )

        writer.add_scalars(
            'data/acc',
            {'train/accuracy': t_acc, 'valid/acc': v_acc},
            epoch + 1
        )

        if max_loss > v_loss:
            torch.save({
                'weights':model.state_dict(),
                'loss':v_loss,
                'epoch':epoch
            }, args.save)

    writer.export_scalars_to_json(args.fmodel + '.json')
    writer.close()

    print('finished learning')

    conmat = test()
    torch.save(model.state_dict(), args.fmodel + '.pth')
    np.savetxt(args.fmodel + 'conf_mat.txt', np.array(conmat), fmt='%s', delimiter=',')