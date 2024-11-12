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

from advertorch.attacks import LinfPGDAttack, GradientSignAttack, L1PGDAttack, L2PGDAttack, DeepfoolLinfAttack
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import os
import random

class AdversarialDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the adversarial images.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))  # Sort to maintain order
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Open and convert to RGB
        label = int(self.image_files[idx].split('_')[-1].split('.')[0])  # Assuming label is encoded in filename


        if self.transform:
            image = self.transform(image)

        return image, label

def create_adversarial_loaders(base_dir, batch_size=128, num_workers=2):
    """
    Args:
        base_dir (string): The base directory containing model-specific adversarial subdirectories.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of worker threads for data loading.
    
    Returns:
        dict: A dictionary of DataLoader objects, keyed by `<model>/<attack>`.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Dictionary to store loaders
    loaders = []

    # Walk through the directory structure
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        print(model_dir)
        if not os.path.isdir(model_dir):
            continue  # Skip non-directory entries
        
        for attack_name in os.listdir(model_dir):
            attack_dir = os.path.join(model_dir, attack_name)
            if not os.path.isdir(attack_dir):
                continue  # Skip non-directory entries
            
            # Create a DataLoader for this attack
            dataset = AdversarialDataset(root_dir=attack_dir, transform=transform)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            loaders.append(loader)

    return loaders

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



base_dir = "./data/perturbed_train"  # Base directory containing adversarial data
train_loaders = create_adversarial_loaders(base_dir)
train_loaders.append(trainloader)
print(len(train_loaders))
for loader in train_loaders:
    print(len(loader))



testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = ResNet18().to(device)

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



import csv
import time
import os
from datetime import datetime

# Create a unique checkpoint directory to avoid overwriting previous runs
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = f'./checkpoint/resnet18_singlemodel_EAT_premade_data_actually20cleanfirst'
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
    
    # Synchronize iterators across all loaders
    iterators = [iter(loader) for loader in train_loaders]
    
    for batch_idx in range(len(train_loaders[0])):  # All loaders have the same length
        # Advance all iterators to maintain synchronization
        batches = [next(iterator) for iterator in iterators]  # Load the current batch from all loaders
        
        # Randomly pick which batch to use for this iteration
        if epoch < 20:
            selected_loader_index = -1
        else:
            selected_loader_index = random.randint(0, len(train_loaders) - 1)
        inputs, targets = batches[selected_loader_index]
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
        
        progress_bar(batch_idx, len(train_loaders[0]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    train_acc = 100. * correct / total
    train_time = time.time() - start_time
    
    return train_acc, train_time

def test(epoch):
    global best_acc

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

    # Save checkpoint for best regular accuracy
    if test_acc > best_acc:
        print('Saving best regular accuracy model..')
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'best_regular_model.pth'))
        best_acc = test_acc

    # Save model after every epoch
    # torch.save(net.state_dict(), os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth'))

    return test_acc, test_time

# Create CSV file with headers
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Epoch', 'Train_Acc', 'Clean_Test_Acc', 'Train_Time', 'Clean_Test_Time'])

for epoch in range(start_epoch, start_epoch + 200):
    train_acc, train_time = train(epoch)
    clean_test_acc, clean_test_time = test(epoch)
    scheduler.step()

    # Write results to CSV
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch, train_acc, clean_test_acc, train_time, clean_test_time])

