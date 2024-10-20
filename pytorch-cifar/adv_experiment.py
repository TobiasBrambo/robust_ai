'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from advertorch.attacks import LinfPGDAttack
import numpy as np


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

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/resnet18_adversarial_training/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# params based on: https://github.com/BorealisAI/advertorch/issues/76#issuecomment-692436644
adversary = LinfPGDAttack(net, criterion, eps=0.031, nb_iter=10 or 7, eps_iter=0.007)

import csv
import time
import os
from datetime import datetime

# Create a unique checkpoint directory to avoid overwriting previous runs
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = f'./checkpoint/resnet18_adversarial_training_{current_time}'
os.makedirs(checkpoint_dir, exist_ok=True)

csv_file_path = os.path.join(checkpoint_dir, 'training_results.csv')

best_acc = 0
best_adv_acc = 0
best_combined_acc = 0

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if epoch >= 20 and np.random.random() < 0.5:
            net.eval()
            inputs = adversary.perturb(inputs, targets)
            net.train()
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

    # Store training results to CSV
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, 'train', train_acc, train_time])

def test(epoch):
    global best_acc, best_adv_acc, best_combined_acc

    # Regular test
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    test_acc = 100. * correct / total
    test_time = time.time() - start_time

    # Store regular test results to CSV
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, 'test', test_acc, test_time])

    # Save checkpoint for best regular accuracy
    if test_acc > best_acc:
        print('Saving best regular accuracy model..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_dir, f'best_regular_ckpt_ep{epoch}_acc{test_acc}.pth'))
        best_acc = test_acc

    # Adversarial test
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Generate adversarial examples outside of torch.no_grad()
        net.train()  # Set to train mode to allow gradients for adversary
        inputs = adversary.perturb(inputs, targets)
        net.eval()  # Switch back to evaluation mode
        
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    adv_test_acc = 100. * correct / total
    test_time = time.time() - start_time

    # Store adversarial test results to CSV
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, 'test_adv', adv_test_acc, test_time])

    # Save checkpoint for best adversarial accuracy
    if adv_test_acc > best_adv_acc:
        print('Saving best adversarial accuracy model..')
        state = {
            'net': net.state_dict(),
            'acc': adv_test_acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_dir, f'best_adv_ckpt_ep{epoch}_acc{adv_test_acc}.pth'))
        best_adv_acc = adv_test_acc

    # Save checkpoint for best combined accuracy (regular + adversarial)
    combined_acc = (test_acc + adv_test_acc) / 2
    if combined_acc > best_combined_acc:
        print('Saving best combined accuracy model..')
        state = {
            'net': net.state_dict(),
            'acc': combined_acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(checkpoint_dir, f'best_combined_ckpt_ep{epoch}_acc{combined_acc}.pth'))
        best_combined_acc = combined_acc

# Create CSV file with headers
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Phase', 'Accuracy', 'Time (s)'])

for epoch in range(start_epoch, start_epoch + 200):
    train(epoch)
    test(epoch)
    scheduler.step()
