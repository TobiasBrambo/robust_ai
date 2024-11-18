import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import torch.backends.cudnn as cudnn
from advertorch.attacks import LinfPGDAttack
from models import ResNet18
import cv2
from scipy.ndimage import median_filter
import torch.nn.functional as F

# Feature Squeezing Functions
def reduce_color_depth_batch(batch, bit_depth=4, new_mean: bool = True):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    batch_unnormalized = batch * torch.tensor(std, device=batch.device).view(1, 3, 1, 1) + torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    batch_unnormalized = torch.clamp(batch_unnormalized, 0, 1)

    scale = 2 ** bit_depth - 1
    batch_bit_reduced = torch.round(batch_unnormalized * scale) / scale

    if new_mean:
        new_mean = batch_bit_reduced.mean().item()
        new_std = batch_bit_reduced.std().item()
        batch_renormalized = (batch_bit_reduced - new_mean) / new_std
    else:
        batch_renormalized = (batch_bit_reduced - torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)) / torch.tensor(std, device=batch.device).view(1, 3, 1, 1)

    return batch_renormalized


def median_filter_batch(batch, filter_size=3):
    batch = (batch * torch.tensor([0.2023, 0.1994, 0.2010], device=batch.device).view(1, 3, 1, 1)) + \
            torch.tensor([0.4914, 0.4822, 0.4465], device=batch.device).view(1, 3, 1, 1)
    batch = torch.clamp(batch, 0, 1)

    batch_np = batch.permute(0, 2, 3, 1).to('cpu').numpy()
    smoothed = np.stack([median_filter(img, size=(filter_size, filter_size, 1)) for img in batch_np])
    smoothed = torch.tensor(smoothed, dtype=torch.float, device=batch.device).permute(0, 3, 1, 2).to(batch.device)

    smoothed = (smoothed - torch.tensor([0.4914, 0.4822, 0.4465], device=batch.device).view(1, 3, 1, 1)) / \
               torch.tensor([0.2023, 0.1994, 0.2010], device=batch.device).view(1, 3, 1, 1)

    return smoothed


def non_local_means_batch(batch, h=10, template_window_size=7, search_window_size=21):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    batch_unnormalized = batch * torch.tensor(std, device=batch.device).view(1, 3, 1, 1) + torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    batch_unnormalized = torch.clamp(batch_unnormalized, 0, 1)

    batch_np = batch_unnormalized.permute(0, 2, 3, 1).cpu().numpy()
    smoothed = []
    for img in batch_np:
        img_uint8 = (img * 255.0).astype(np.uint8)
        smoothed_img = cv2.fastNlMeansDenoisingColored(
            img_uint8, None, h, h, template_window_size, search_window_size
        )
        smoothed.append(smoothed_img / 255.0)
    smoothed = np.stack(smoothed)
    smoothed_tensor = torch.tensor(smoothed, dtype=torch.float, device=batch.device).permute(0, 3, 1, 2)

    batch_renormalized = (smoothed_tensor - torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)) / torch.tensor(std, device=batch.device).view(1, 3, 1, 1)
    return batch_renormalized


def calculate_optimal_threshold(model, loader, squeezer, adversarial_attack=None, squeezer_name="squeezer"):
    """
    Calculate the optimal threshold for detecting adversarial examples.

    Parameters:
        model: Trained neural network model.
        loader: DataLoader containing legitimate and adversarial examples.
        squeezer: Feature squeezing function to be applied.
        adversarial_attack: Attack function to generate adversarial examples.
        squeezer_name: Name of the squeezer for saving the plot.

    Returns:
        optimal_threshold: Optimal threshold value for detecting adversarial examples.
    """
    model.eval()
    device = "cuda"
    all_distances_legitimate = []
    all_distances_adversarial = []
    count = 0
    m = nn.Softmax()

    for images, true_labels in loader:
        do_adv = 0 if np.random.random() < 0.5 else 1
        count += 1
        images = images.to(device)
        true_labels = true_labels.to(device)
        if do_adv:
            images.requires_grad = True
            images = adversarial_attack.perturb(images, true_labels)

        with torch.no_grad():
            if do_adv:
                original_prediction =m(model(images))
                squeezed_images = squeezer(images)
                squeezed_prediction = m(model(squeezed_images))
                batch_distances = torch.nn.functional.l1_loss(
                    original_prediction, squeezed_prediction, reduction='none'
                ).view(images.size(0), -1).sum(dim=1)
                all_distances_adversarial.extend(batch_distances.cpu().numpy())
            else:
                original_prediction = m(model(images))
                squeezed_images = squeezer(images)
                squeezed_prediction = m(model(squeezed_images))
                batch_distances = torch.nn.functional.l1_loss(
                    original_prediction, squeezed_prediction, reduction='none'
                ).view(images.size(0), -1).sum(dim=1)
                all_distances_legitimate.extend(batch_distances.cpu().numpy())
        if count == 100:
            break

    # Plot histograms to visualize the separation between legitimate and adversarial examples
    plt.figure()
    plt.hist(all_distances_legitimate, bins=50, alpha=0.5, label='Legitimate Examples', color='b')
    plt.hist(all_distances_adversarial, bins=50, alpha=0.5, label='Adversarial Examples', color='r')
    plt.xlabel('L1 Distance')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of L1 Distances for Legitimate and Adversarial Examples ({squeezer_name})')
    plt.legend(loc='upper right')
    plt.savefig(f'histogram_l1_distances_{squeezer_name}.png')
    # print(all_distances_legitimate)
    # print(all_distances_adversarial)

    # Determine optimal threshold by visually inspecting the histogram or using a heuristic
    optimal_threshold = (np.mean(all_distances_legitimate) + np.mean(all_distances_adversarial)) / 2

    return optimal_threshold

# Load model and DataLoader
device = "cuda"
model = ResNet18().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

model.load_state_dict(torch.load("checkpoint/resnet18_clean/best_regular_model.pth"))

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_loader = DataLoader(
    torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),
    batch_size=1, shuffle=False, num_workers=2
)

# Define adversarial attack
adversarial_attack = LinfPGDAttack(model, loss_fn=F.cross_entropy, eps=0.05, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3)

# Define feature squeezers with their respective names
squeezers = [
    (lambda batch: reduce_color_depth_batch(batch, bit_depth=4), "reduce_color_depth_4bit"),
    (lambda batch: reduce_color_depth_batch(batch, bit_depth=5), "reduce_color_depth_5bit"),
    (lambda batch: median_filter_batch(batch, filter_size=2), "median_filter_2x2"),
    (lambda batch: median_filter_batch(batch, filter_size=3), "median_filter_3x3"),
    (lambda batch: non_local_means_batch(batch, h=2, template_window_size=3, search_window_size=11), "nlm_h2_tw3_sw11"),
    (lambda batch: non_local_means_batch(batch, h=4, template_window_size=3, search_window_size=11), "nlm_h4_tw3_sw11"),
    (lambda batch: non_local_means_batch(batch, h=2, template_window_size=3, search_window_size=13), "nlm_h2_tw3_sw13"),
    (lambda batch: non_local_means_batch(batch, h=4, template_window_size=3, search_window_size=13), "nlm_h4_tw3_sw13")
]

# Calculate optimal threshold for each squeezer separately
for squeezer, squeezer_name in squeezers:
    optimal_threshold = calculate_optimal_threshold(model, test_loader, squeezer, adversarial_attack=adversarial_attack, squeezer_name=squeezer_name)
    print(f"Optimal Threshold for {squeezer_name}: {optimal_threshold}")

