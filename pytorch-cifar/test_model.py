
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

from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, SparseL1DescentAttack
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

# Load checkpoint.

# params based on: https://github.com/BorealisAI/advertorch/issues/76#issuecomment-692436644

import time
import os
import csv

def test_LinfPGD(net, criterion):
    pass

def test_SparseL1Descent(net, criterion):
    pass

def test_xyz():
    pass




def loop_model_checkpoints(directory):

    net = ResNet18()
    
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    files = os.listdir(directory)

    # Prepare CSV file for results
    csv_file = os.path.join(directory, "test_results.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Regular Accuracy (%)", "LinfPGD Accuracy (%)", "SparseL1Descent Accuracy (%)"])

        for epoch in range(200):
            file_pattern = f'model_epoch_{epoch}.pth'
            if file_pattern in files:
                checkpoint_path = os.path.join(directory, file_pattern)
                print(f'Testing checkpoint: {file_pattern}')
                
                # Load the checkpoint
                checkpoint = torch.load(checkpoint_path)
                net.load_state_dict(checkpoint)
                
                # Define adversarial attacks
                criterion = nn.CrossEntropyLoss()
                adversary = LinfPGDAttack(net, criterion, eps=0.031, nb_iter=10, eps_iter=0.007)
                adversary_2 = SparseL1DescentAttack(net, criterion) 
                
                # Regular test
                net.eval()
                test_loss = 0
                correct = 0
                total = 0
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
                regular_acc = 100. * correct / total

                # Adversarial test 1
                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
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
                adv_acc_1 = 100. * correct / total

                # Adversarial test 2
                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs = adversary_2.perturb(inputs, targets)
                    
                    with torch.no_grad():
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)

                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                adv_acc_2 = 100. * correct / total

                # Write results to CSV
                writer.writerow([epoch, regular_acc, adv_acc_1, adv_acc_2])

loop_model_checkpoints("./checkpoint/resnet18_adversarial_training_20241021_215311/")

