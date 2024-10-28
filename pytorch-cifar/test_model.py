
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

from advertorch.attacks import LinfPGDAttack, CarliniWagnerL2Attack, SparseL1DescentAttack, DeepfoolLinfAttack, L2PGDAttack, SpatialTransformAttack
import numpy as np
import time
import os
import csv


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
net = ResNet18()





def reset_bn_stats(model, dataloader, device):
    model.train()  # Set to train mode to update BN stats
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)  # Forward pass to update BN running mean/variance

    model.eval()  # Return model to eval mode for testing




def loop_model_checkpoints(directory):

    net = ResNet18()
    
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    # Prepare CSV file for results
    csv_file = os.path.join(directory, "test_results.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Regular Accuracy (%)", "LinfPGD Accuracy (%)", "DeepfoolLinf Accuracy (%)"])

        files = ["best_regular_model.pth", "best_combined_model.pth", "best_adv_model.pth"]
        # files = ["best_regular_model.pth"]


        for file in files:
            checkpoint_path = os.path.join(directory, file)
            print(f'Testing checkpoint: {file}')
           
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint)
            
            reset_bn_stats(net, trainloader, device)
            net.eval()
            
            # Define adversarial attacks
            criterion = nn.CrossEntropyLoss()
            adversary = LinfPGDAttack(net, criterion, eps=0.05, nb_iter=20, eps_iter=0.01)
            adversary_2 = DeepfoolLinfAttack(net, 10, nb_iter=50, eps=0.031, loss_fn=criterion)
            # adversary_2 = L2PGDAttack(net, criterion, eps = 0.031, nb_iter=10, eps_iter=0.007)
            # adversary = SpatialTransformAttack(net, num_classes=10, loss_fn=criterion)
            
            # Regular test
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
                net.train()
                inputs = adversary_2.perturb(inputs, targets)
                net.eval()
                
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
            writer.writerow([file, regular_acc, adv_acc_1, adv_acc_2])

# loop_model_checkpoints("./checkpoint/resnet18_clean/")
loop_model_checkpoints("./checkpoint/resnet18_adversarial_training_gradientsign_20241027_173813/")
# loop_model_checkpoints("./checkpoint/resnet18_clean_20241024_071233/")

# benchmark_resnet18_advtrain()
