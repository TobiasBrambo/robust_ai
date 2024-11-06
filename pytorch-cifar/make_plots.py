import os
import matplotlib.pyplot as plt
from advertorch.attacks import (
    GradientSignAttack,
    LinfPGDAttack,
    SinglePixelAttack,
    L2PGDAttack,
    DeepfoolLinfAttack,
)
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import ResNet18

# Create 'photos' folder if it doesn't exist
os.makedirs("photos", exist_ok=True)

# Preprocessing
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Dataset and DataLoader
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=0)

# Load model
net = ResNet18()
device = "cpu"
net = net.to(device)
checkpoint_path = "checkpoint/resnet18_clean/best_regular_model.pth"
state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v
net.load_state_dict(new_state_dict)
net.eval()

# Define adversarial attacks
criterion = nn.CrossEntropyLoss()
adversaries = {
    "LinfPGD": LinfPGDAttack(net, criterion, eps=0.05, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3),
    "FGSM": GradientSignAttack(net, criterion, eps=0.05, clip_min=-3, clip_max=3),
    "DeepFool": DeepfoolLinfAttack(net, 10, nb_iter=10, eps=0.05, loss_fn=criterion, clip_min=-3, clip_max=3),
    "SinglePixelAttack": SinglePixelAttack(net, loss_fn=criterion, max_pixels=32*32, clip_min=-3, clip_max=3),
    # "L2PGD": L2PGDAttack(net, criterion, eps=0.05, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3),
}

# Find suitable images for each adversarial attack
selected_images = {}  # To store the images, labels, and adversarial results for each method
data_iter = iter(trainloader)
for name, adversary in adversaries.items():
    suitable_image_found = False

    print(name)
    print()
    
    # data_iter = iter(trainloader)
    count = 0
    while not suitable_image_found:
        print(count)
        count += 1
        # Get the next image and label
        if count%32 == 0:
            data_iter = iter(trainloader)

        image, target = next(data_iter)
        first_image, first_label = image[0].unsqueeze(0).to(device), target[0].item()

        # Classify the original image
        clean_output = net(first_image)  # Model output for the clean image
        clean_pred = clean_output.argmax(dim=1).item()
        if clean_pred != first_label:
            print("model bad")
            continue  # Skip if the clean image is already misclassified

        # Generate the adversarial example
        adv_image = adversary.perturb(first_image, torch.tensor([first_label]).to(device))
        adv_output = net(adv_image)
        adv_pred = adv_output.argmax(dim=1).item()

        # Check if the adversarial example is misclassified
        if adv_pred != first_label:
            # Save the suitable image, label, and adversarial result
            selected_images[name] = {
                "original_image": first_image,
                "original_label": first_label,
                "adversarial_image": adv_image,
                "adversarial_pred": adv_pred,
            }
            suitable_image_found = True

# Initialize the grid layout (3x5 grid: original, perturbed, heatmap for each method)
# Initialize the grid layout (3x5 grid: original, perturbed, heatmap for each method)
fig, axes = plt.subplots(len(adversaries), 3, figsize=(15, 5 * len(adversaries)))

mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)

for i, (name, data) in enumerate(selected_images.items()):
    original_image = data["original_image"]
    original_label = data["original_label"]
    adversarial_image = data["adversarial_image"]
    adversarial_pred = data["adversarial_pred"]

    # Denormalize images for visualization
    original_image_denorm = (original_image * std + mean).clamp(0, 1)
    adversarial_image_denorm = (adversarial_image * std + mean).clamp(0, 1)
    original_image_np = original_image_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adversarial_image_np = adversarial_image_denorm.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Compute perturbation
    perturbation = adversarial_image_np - original_image_np
    perturbation_scaled = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)

    # Get confidence for the original label
    clean_output = net(original_image)  # Model output for the clean image
    adv_output = net(adversarial_image)  # Model output for the adversarial image

    clean_confidence = torch.softmax(clean_output, dim=1)[0, original_label].item() * 100
    adv_confidence = torch.softmax(adv_output, dim=1)[0, original_label].item() * 100

    # Plot original image
    axes[i, 0].imshow(original_image_np)
    axes[i, 0].set_title(
        f"Original Image\nLabel: {trainset.classes[original_label]}\nConfidence: {clean_confidence:.2f}%"
    )
    axes[i, 0].axis("off")

    # Plot adversarial image
    axes[i, 1].imshow(adversarial_image_np)
    axes[i, 1].set_title(
        f"Adversarial ({name})\nPredicted: {trainset.classes[adversarial_pred]}\nConfidence for Original Label: {adv_confidence:.2f}%"
    )
    axes[i, 1].axis("off")

    # Plot perturbation heatmap
    axes[i, 2].imshow(perturbation_scaled)
    axes[i, 2].set_title("Perturbation Heatmap")
    axes[i, 2].axis("off")

# Adjust layout and save the grid
plt.tight_layout()
plt.savefig("photos/adversarial_grid.pdf", dpi=300, format="pdf")
plt.close()

print("Adversarial examples and heatmaps saved in a grid layout.")
