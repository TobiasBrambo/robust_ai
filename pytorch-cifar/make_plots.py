

import csv
import os
import os

from advertorch.attacks import GradientSignAttack, LinfPGDAttack, SinglePixelAttack, L2PGDAttack, DeepfoolLinfAttack
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import *
from utils import progress_bar


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


net = ResNet18()


device = "cuda"
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True



# files = ["best_regular_model.pth"]

directory = "checkpoint/resnet18_clean"
file = "best_regular_model.pth"

checkpoint_path = os.path.join(directory, file)
print(f'Testing checkpoint: {file}')

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint)

reset_bn_stats(net, trainloader, device)
net.eval()

# Define adversarial attacks
criterion = nn.CrossEntropyLoss()
adversary = LinfPGDAttack(net, criterion, eps=0.05, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3)
adversary_2 = GradientSignAttack(net, criterion, clip_min=-3, clip_max=3)
adversary_3 = DeepfoolLinfAttack(net, 10, nb_iter=10, eps=0.05, loss_fn=criterion, clip_min=-3, clip_max=3)
adversary_4 = GradientSignAttack(net, criterion, eps=0.05, clip_min=-3, clip_max=3)
adversary_5 = L2PGDAttack(net, criterion, eps=0.05, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3)
adversary_6 = SinglePixelAttack(net, loss_fn=criterion, max_pixels=1, clip_min=-3, clip_max=3)
adversary_7 = SinglePixelAttack(net, loss_fn=criterion, max_pixels=10, clip_min=-3, clip_max=3)
adversary_8 = LinfPGDAttack(net, criterion, eps=0.3, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3)


data_iter = iter(trainloader)
image, target = next(data_iter)

first_image, first_target = image[0], target[0]



