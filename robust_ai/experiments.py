from data.cifar10 import get_cifar10

from .models import CNN
# from dependencies.visualizer import MultiClassVisualizer
import numpy as np 
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import tqdm 
import warnings
# from dependencies import suppresser


import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cnn_cifar10_experiment():

    train_loader, test_loader, classes = get_cifar10(n_classes=0, batch_size=128)


    loss_func = torch.nn.CrossEntropyLoss()
    model = CNN().to(device)

    model.fit(train_loader, test_loader, loss_function=loss_func, num_epochs=100)



