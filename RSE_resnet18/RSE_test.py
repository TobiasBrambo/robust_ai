
import argparse
import csv
from datetime import datetime
import os
import os
import time

from advertorch.attacks import DeepfoolLinfAttack, GradientSignAttack, L2PGDAttack, LinfPGDAttack, SinglePixelAttack
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from resnet_RSE import ResNet18
from utils import progress_bar


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

criterion = nn.CrossEntropyLoss()
# params based on: https://github.com/BorealisAI/advertorch/issues/76#issuecomment-692436644
# adversary = LinfPGDAttack(net, criterion, eps=0.031, nb_iter=10, eps_iter=0.007)
# adversary = LinfPGDAttack(net, criterion, eps=0.04, nb_iter=10, eps_iter=0.007)
# adversary = GradientSignAttack(net, criterion)



# Create a unique checkpoint directory to avoid overwriting previous runs
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = f'./checkpoint/resnet18_RSE'
os.makedirs(checkpoint_dir, exist_ok=True)

csv_file_path = os.path.join(checkpoint_dir, 'training_results.csv')

best_acc = 0
best_adv_acc = 0
best_combined_acc = 0




def reset_bn_stats(model, dataloader, device):
    model.train()  # Set to train mode to update BN stats
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)  # Forward pass to update BN running mean/variance

    model.eval()  # Return model to eval mode for testing



def test_loop(net, adversary = None, num_tests_per_batch:int = 10, set_train_for_gen: bool = True):

    test_loss = 0
    correct = 0
    total = 0
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        if adversary:
            if set_train_for_gen:
                with torch.enable_grad():
                    inputs = adversary.perturb(inputs, targets)
            else:
                inputs = adversary.perturb(inputs, targets)

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
    regular_acc = 100. * correct / total
    return regular_acc




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

        # files = ["best_regular_model.pth"]

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

        num_tests_per_batch = 10

        writer.writerow(["Epoch", "Regular Accuracy (%)", 
                         "LinfPGD  eps005 nbiter10 epsiter 001 Accuracy (%)",
                         "FGSM  default params Accuracy (%)", 
                         "DeepfoolLinf eps005 nbiter10 Accuracy (%)", 
                         "FGSM eps005 Accuracy (%)",
                         "L2PGD eps005 nbiter10 epsiter001 Accuracy (%)",
                         "SinglePixelAttack maxpixels1 Accuracy (%)",
                         "SinglePixelAttack maxpixels10 Accuracy (%)"])
        
        regular_acc = test_loop(net, adversary=None, num_tests_per_batch=num_tests_per_batch)

        adv_acc_1 = test_loop(net, adversary=adversary, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        adv_acc_2 = test_loop(net, adversary=adversary_2, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        adv_acc_3 = test_loop(net, adversary=adversary_3, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        adv_acc_4 = test_loop(net, adversary=adversary_4, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        adv_acc_5 = test_loop(net ,adversary=adversary_5, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        adv_acc_6 = test_loop(net, adversary=adversary_6, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        adv_acc_7 = test_loop(net, adversary=adversary_7, num_tests_per_batch=num_tests_per_batch, set_train_for_gen=True)
        

        # Write results to CSV
        writer.writerow([file, regular_acc, adv_acc_1, adv_acc_2, adv_acc_3, adv_acc_4, adv_acc_5, adv_acc_6, adv_acc_7])


loop_model_checkpoints("checkpoint/resnet18_RSE")
