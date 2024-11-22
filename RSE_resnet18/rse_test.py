'''Train CIFAR10 with PyTorch.'''
import argparse
import csv
from datetime import datetime
import os
import os
import time

from advertorch.attacks import GradientSignAttack, LinfPGDAttack
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar




from torchvision.models import resnet18
from pytorch_rse.rse import RSE
device = "cuda"
model = resnet18(num_classes=10)
net = RSE(model, std_devs=(0.1,0.1))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100. * correct / total
    train_time = time.time() - start_time

    return train_acc, train_time

def test(epoch):
    global best_acc, best_adv_acc, best_combined_acc

    # Regular test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_tests_per_batch = 5

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        accumulated_probs = torch.zeros(inputs.size(0), 10).to(device)
        temp_loss = 0
        prev_outputs = None

        for _ in range(num_tests_per_batch):
            with torch.no_grad():
                outputs = net(inputs)
                if prev_outputs != None:
                    assert not outputs.equal(prev_outputs)
                prev_outputs = outputs
                temp_loss += criterion(outputs, targets).item()
                probs = F.softmax(outputs, dim=1)
                accumulated_probs += probs



        test_loss += (temp_loss/num_tests_per_batch)
        _, predicted = accumulated_probs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100. * correct / total
    test_time = time.time() - start_time

for epoch in range(100):
    train_acc, train_time = train(epoch)
    test(epoch)
    scheduler.step()
