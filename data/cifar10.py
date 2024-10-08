import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
_LOGGER = logging.getLogger(__name__)

def get_cifar10(batch_size=64, n_classes=5, seed=np.random.randint(1, 100)):
    _LOGGER.info("Generalized Unbalanced CIFAR-10: Preparing dataset...")

    random.seed(seed)
    np.random.seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/raw_files', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data/raw_files', train=False, download=True, transform=transform)

    targets = np.array(trainset.targets)
    classes = np.unique(targets)

    initial_distribution = {trainset.classes[i]: np.sum(targets == i) for i in range(10)}

    imbalanced_classes = np.random.choice(classes, n_classes, replace=False)

    indices_to_keep = []
    for class_label in classes:
        class_indices = np.where(targets == class_label)[0]
        if class_label in imbalanced_classes:

            remove_fraction = random.uniform(0.5, 0.9)
            num_to_keep = int(len(class_indices) * (1 - remove_fraction))
        else:
            num_to_keep = len(class_indices)
        
        indices_to_keep.extend(np.random.choice(class_indices, num_to_keep, replace=False))

    unbalanced_trainset = Subset(trainset, indices_to_keep)

    train_loader = torch.utils.data.DataLoader(unbalanced_trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


    new_targets = np.array([trainset.targets[i] for i in indices_to_keep])

    final_distribution = {trainset.classes[i]: np.sum(new_targets == i) for i in range(10)}
    _LOGGER.info("Class distribution:")
    for class_name, count in final_distribution.items():
        initial_count = initial_distribution[class_name]
        difference = initial_count - count

        percentage = 100 - (difference / initial_count) * 100 if initial_count > 0 else 0

        _LOGGER.info(f"{class_name}: {count} ({percentage:.2f}%)")

    _LOGGER.info("Generalized Unbalanced CIFAR-10: Preparing dataset...\t DONE")

    return train_loader, test_loader, trainset.classes


if __name__ == '__main__':

    train_loader, test_loader, classes = get_cifar10(batch_size=64, n_classes=3, seed=42)
