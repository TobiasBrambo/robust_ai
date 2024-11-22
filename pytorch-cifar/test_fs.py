import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2

from feature_squeezing import (
    median_filter_batch,
    non_local_means_batch
)

# Load one clean image from CIFAR-10 dataset
def load_clean_image():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    image, label = next(iter(loader))
    return image, label

# Modified NLM function with debug prints
def non_local_means_batch_debug(batch, h=10, template_window_size=7, search_window_size=21):
    # Unnormalize the image
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    batch_unnormalized = batch * torch.tensor(std, device=batch.device).view(1, 3, 1, 1) + torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    batch_unnormalized = torch.clamp(batch_unnormalized, 0, 1)

    print("Debug: Unnormalized batch min:", batch_unnormalized.min().item(), "max:", batch_unnormalized.max().item())

    batch_np = batch_unnormalized.permute(0, 2, 3, 1).cpu().numpy()
    smoothed = []
    for img in batch_np:
        img_uint8 = (img * 255.0).astype(np.uint8)
        print("Debug: Image before NLM min:", img_uint8.min(), "max:", img_uint8.max())
        smoothed_img = cv2.fastNlMeansDenoisingColored(
            img_uint8, None, h, h, template_window_size, search_window_size
        )
        print("Debug: Image after NLM min:", smoothed_img.min(), "max:", smoothed_img.max())
        smoothed.append(smoothed_img / 255.0)
    smoothed = np.stack(smoothed)
    smoothed_tensor = torch.tensor(smoothed, dtype=torch.float).permute(0, 3, 1, 2).to(batch.device)

    # Renormalize the image
    batch_renormalized = (smoothed_tensor - torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)) / torch.tensor(std, device=batch.device).view(1, 3, 1, 1)
    print("Debug: Renormalized batch min:", batch_renormalized.min().item(), "max:", batch_renormalized.max().item())

    return batch_renormalized

# Save original and squeezed images side by side
def save_and_display_squeezed_images(image, squeezed_images, titles, save_path="squeezed_images.png"):
    plt.figure(figsize=(15, 10))
    num_images = len(squeezed_images) + 1

    # Original image
    plt.subplot(1, num_images, 1)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title("Original Image")
    plt.axis('off')

    # Squeezed images
    for i, squeezed_image in enumerate(squeezed_images):
        plt.subplot(1, num_images, i + 2)
        plt.imshow(squeezed_image.permute(1, 2, 0).cpu().numpy())
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Squeezed images saved to {save_path}")

def reduce_color_depth_batch_debug(batch, bit_depth=4):
    # Unnormalize the image
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    batch_unnormalized = batch * torch.tensor(std, device=batch.device).view(1, 3, 1, 1) + torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    batch_unnormalized = torch.clamp(batch_unnormalized, 0, 1)

    print("Debug: Unnormalized batch min:", batch_unnormalized.min().item(), "max:", batch_unnormalized.max().item())

    # Reduce bit depth
    scale = 2 ** bit_depth - 1
    batch_bit_reduced = torch.round(batch_unnormalized * scale) / scale

    # Calculate new mean and std for renormalization
    new_mean = batch_bit_reduced.mean().item()
    new_std = batch_bit_reduced.std().item()
    print("Debug: Bit-reduced batch mean:", new_mean, "std:", new_std)

    # Renormalize the image
    batch_renormalized = (batch_bit_reduced - new_mean) / new_std
    print("Debug: Renormalized batch min:", batch_renormalized.min().item(), "max:", batch_renormalized.max().item())

    return batch_bit_reduced, new_mean, new_std


# Main function to run the feature squeezing and visualization
def main():
    # Load the clean image
    image, label = load_clean_image()
    image = image.cuda()

    # Apply feature squeezers
    squeezed_images = []
    titles = []

    # Reduce color depth with new function
    # squeezed_images.append(reduce_color_depth_batch_debug(image, bit_depth=4).squeeze())
    bit_reduced_batch, new_mean, new_std = reduce_color_depth_batch_debug(image, bit_depth=4)
    batch_new_normalized = (bit_reduced_batch - new_mean) / new_std
    squeezed_images.append(batch_new_normalized.squeeze())
    titles.append("4-bit Depth (new Batch Renormalized)")

    old_mean = [0.4914, 0.4822, 0.4465]
    old_std = [0.2023, 0.1994, 0.2010]
    # torch.tensor(old_std, device=batch.device).view(1, 3, 1, 1) + torch.tensor(old_mean, device=batch.device).view(1, 3, 1, 1)
    batch_old_normalized = (bit_reduced_batch - torch.tensor(old_mean, device="cuda").view(1, 3, 1, 1)) / torch.tensor(old_std, device="cuda").view(1, 3, 1, 1) 
    squeezed_images.append(batch_old_normalized.squeeze())
    titles.append("4-bit Depth (old Batch Renormalized)")

    # Apply median filter 3x3
    squeezed_images.append(median_filter_batch(image, filter_size=3).squeeze())
    titles.append("Median 3x3")

    # Apply non-local means filter with different values for debugging
    nlm_params = [
        (11, 3, 2),
        (11, 3, 4),
        (13, 3, 2),
        (13, 3, 4)
    ]
    for search_window, template_window, h in nlm_params:
        squeezed_images.append(non_local_means_batch_debug(image, h=h, template_window_size=template_window, search_window_size=search_window).squeeze())
        titles.append(f"NLM {search_window}-{template_window}-{h}")

    # Save images side by side
    save_and_display_squeezed_images(image.squeeze(), squeezed_images, titles)

if __name__ == "__main__":
    main()

