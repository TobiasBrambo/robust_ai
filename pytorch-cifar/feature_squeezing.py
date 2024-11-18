import csv
from tqdm import tqdm
import numpy as np
import cv2
from scipy.ndimage import median_filter

from advertorch.attacks import (
    DeepfoolLinfAttack,
    GradientSignAttack,
    LinfPGDAttack,
)
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np

from models import ResNet18

# Feature Squeezing Functions
def reduce_color_depth_batch(batch, bit_depth=4):
    # Unnormalize the image
    batch = (batch * torch.tensor([0.2023, 0.1994, 0.2010], device=batch.device).view(1, 3, 1, 1)) + \
            torch.tensor([0.4914, 0.4822, 0.4465], device=batch.device).view(1, 3, 1, 1)
    batch = torch.clamp(batch, 0, 1)

    # Reduce bit depth
    scale = 2 ** bit_depth - 1
    batch = torch.round(batch * scale) / scale

    # Renormalize the image
    batch = (batch - torch.tensor([0.4914, 0.4822, 0.4465], device=batch.device).view(1, 3, 1, 1)) / \
            torch.tensor([0.2023, 0.1994, 0.2010], device=batch.device).view(1, 3, 1, 1)

    return batch

def median_filter_batch(batch, filter_size=3):
    # Unnormalize the image
    batch = (batch * torch.tensor([0.2023, 0.1994, 0.2010], device=batch.device).view(1, 3, 1, 1)) + \
            torch.tensor([0.4914, 0.4822, 0.4465], device=batch.device).view(1, 3, 1, 1)
    batch = torch.clamp(batch, 0, 1)

    batch_np = batch.permute(0, 2, 3, 1).cpu().numpy()
    smoothed = np.stack([median_filter(img, size=(filter_size, filter_size, 1)) for img in batch_np])
    smoothed = torch.tensor(smoothed, dtype=torch.float).permute(0, 3, 1, 2).to(batch.device)

    # Renormalize the image
    smoothed = (smoothed - torch.tensor([0.4914, 0.4822, 0.4465], device=batch.device).view(1, 3, 1, 1)) / \
               torch.tensor([0.2023, 0.1994, 0.2010], device=batch.device).view(1, 3, 1, 1)

    return smoothed

def non_local_means_batch(batch, h=10, template_window_size=7, search_window_size=21):
    batch_np = batch.permute(0, 2, 3, 1).cpu().numpy()
    smoothed = []
    for img in batch_np:
        # Unnormalize the image
        img = (img * np.array([0.2023, 0.1994, 0.2010])) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        smoothed_img = cv2.fastNlMeansDenoisingColored(
            img, None, h, h, template_window_size, search_window_size
        )
        smoothed.append(smoothed_img / 255.0)
    smoothed = np.stack(smoothed)
    # Renormalize the image
    smoothed = (smoothed / 255.0 - np.array([0.4914, 0.4822, 0.4465])) / np.array([0.2023, 0.1994, 0.2010])
    return torch.tensor(smoothed, dtype=torch.float).permute(0, 3, 1, 2).to(batch.device)

# Feature Squeezing Pipeline
def feature_squeezing_pipeline(model, input_batch, squeezers, threshold):
    model.eval()
    distances = []
    with torch.no_grad():
        original_prediction = model(input_batch)
        for squeezer in squeezers:
            squeezed_batch = squeezer(input_batch)
            squeezed_prediction = model(squeezed_batch)
            batch_distances = F.l1_loss(original_prediction, squeezed_prediction, reduction='none').view(input_batch.size(0), -1).sum(dim=1)
            distances.append(batch_distances)
    max_distances = torch.stack(distances, dim=1).max(dim=1).values
    is_adversarial = max_distances > threshold
    return is_adversarial, max_distances

# Evaluation Function
def evaluate_with_feature_squeezing_grouped(model, loader, squeezers, threshold, adversarial_attack=None, evaluate_clean=False):
    results = {
        "total": 0,
        "correct_legitimate": 0,
        "adversarial_detected": 0,
    }
    
    for images, labels in tqdm(loader, desc="Evaluating batches"):
        images, labels = images.cuda(), labels.cuda()
        
        if adversarial_attack:
            images.requires_grad = True
            images = adversarial_attack.perturb(images, labels)
        
        if evaluate_clean:
            with torch.no_grad():
                squeezed_images = images
                for squeezer in squeezers:
                    squeezed_images = squeezer(squeezed_images)
                predictions = model(squeezed_images)
                _, predicted_labels = predictions.max(1)
                results["correct_legitimate"] += (predicted_labels == labels).sum().item()
                results["total"] += images.size(0)
        else:
            is_adversarial, combined_distances = feature_squeezing_pipeline(model, images, squeezers, threshold)
            
            with torch.no_grad():
                original_predictions = model(images)
                _, predicted_labels = original_predictions.max(1)
            
            results["correct_legitimate"] += ((~is_adversarial) & (predicted_labels == labels)).sum().item()
            results["adversarial_detected"] += is_adversarial.sum().item()
            results["total"] += images.size(0)

    if evaluate_clean:
        results["legitimate_accuracy"] = results["correct_legitimate"] / results["total"]
        results["adversarial_detection_rate"] = "N/A"
    else:
        try:
            results["legitimate_accuracy"] = results["correct_legitimate"] / (results["total"] - results["adversarial_detected"])
        except ZeroDivisionError:
            results["legitimate_accuracy"] = 0.0
        results["adversarial_detection_rate"] = results["adversarial_detected"] / results["total"]
    
    return results

# Usage Example
# Define individual feature squeezers and thresholds for testing individually
individual_squeezers = [
    # {"squeezer": [lambda batch: reduce_color_depth_batch(batch, bit_depth=1)], "threshold": 1.9997, "name": "1-bit"},
    # {"squeezer": [lambda batch: reduce_color_depth_batch(batch, bit_depth=2)], "threshold": 1.9967, "name": "2-bit"},
    # {"squeezer": [lambda batch: reduce_color_depth_batch(batch, bit_depth=3)], "threshold": 1.7822, "name": "3-bit"},
    {"squeezer": [lambda batch: reduce_color_depth_batch(batch, bit_depth=4)], "threshold": 0.7930, "name": "4-bit"},
    {"squeezer": [lambda batch: reduce_color_depth_batch(batch, bit_depth=5)], "threshold": 0.3301, "name": "5-bit"},
    {"squeezer": [lambda batch: median_filter_batch(batch, filter_size=2)], "threshold": 1.1296, "name": "Median 2x2"},
    {"squeezer": [lambda batch: median_filter_batch(batch, filter_size=3)], "threshold": 1.9431, "name": "Median 3x3"},
    {"squeezer": [lambda batch: non_local_means_batch(batch, h=2, template_window_size=3, search_window_size=11)], "threshold": 0.2770, "name": "NLM 11-3-2"},
    {"squeezer": [lambda batch: non_local_means_batch(batch, h=4, template_window_size=3, search_window_size=11)], "threshold": 0.7537, "name": "NLM 11-3-4"},
    {"squeezer": [lambda batch: non_local_means_batch(batch, h=2, template_window_size=3, search_window_size=13)], "threshold": 0.2910, "name": "NLM 13-3-2"},
    {"squeezer": [lambda batch: non_local_means_batch(batch, h=4, template_window_size=3, search_window_size=13)], "threshold": 0.8290, "name": "NLM 13-3-4"},
]

# Define feature squeezers and thresholds for combination testing
squeezer_combination = {
    "squeezers": [
        lambda batch: reduce_color_depth_batch(batch, bit_depth=5),
        lambda batch: median_filter_batch(batch, filter_size=2),
        lambda batch: non_local_means_batch(batch, h=2, template_window_size=3, search_window_size=13)
    ],
    "threshold": 1.1402
}

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
    batch_size=100, shuffle=False, num_workers=2
)

if __name__ == "__main__":
    with open("feature_squeezing_results_fixed_normalization.csv", mode="w", newline="") as csvfile:
        fieldnames = ["Squeezer", "Adversarial Attack", "Legitimate Accuracy", "Adversarial Detection Rate"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Define adversarial attacks
        adversarial_attacks = [
            ("LinfPGD", LinfPGDAttack(model, eps=0.05, nb_iter=10, eps_iter=0.01, clip_min=-3, clip_max=3)),
            ("Deepfool", DeepfoolLinfAttack(model, loss_fn=F.cross_entropy, eps=0.05, nb_iter=10, clip_min=-3, clip_max=3)),
            ("FGSM", GradientSignAttack(model, loss_fn=F.cross_entropy, clip_min=-3, clip_max=3)),
        ]

        # Evaluate each squeezer individually
        for squeezer_info in individual_squeezers:
            # Evaluate on clean dataset without adversarial attack
            print(f"Evaluating {squeezer_info['name']} without adversarial attack")
            results = evaluate_with_feature_squeezing_grouped(
                model, 
                test_loader, 
                squeezer_info["squeezer"], 
                squeezer_info["threshold"],
                adversarial_attack=None,
                evaluate_clean=True
            )

            # Print results
            print(f"{squeezer_info['name']} Results without attack:")
            print("Legitimate Accuracy:", results["legitimate_accuracy"])

            # Write to CSV
            writer.writerow({
                "Squeezer": squeezer_info["name"],
                "Adversarial Attack": "None",
                "Legitimate Accuracy": results["legitimate_accuracy"],
                "Adversarial Detection Rate": "N/A"
            })

            # Evaluate with adversarial attacks
            for attack_name, adversarial_attack in adversarial_attacks:
                print(f"Evaluating {squeezer_info['name']} with {attack_name} attack")
                results = evaluate_with_feature_squeezing_grouped(
                    model, 
                    test_loader, 
                    squeezer_info["squeezer"], 
                    squeezer_info["threshold"],
                    adversarial_attack=adversarial_attack
                )

                # Print results
                print(f"{squeezer_info['name']} Results with {attack_name}:")
                print("Legitimate Accuracy:", results["legitimate_accuracy"])
                print("Adversarial Detection Rate:", results["adversarial_detection_rate"])

                # Write to CSV
                writer.writerow({
                    "Squeezer": squeezer_info["name"],
                    "Adversarial Attack": attack_name,
                    "Legitimate Accuracy": results["legitimate_accuracy"],
                    "Adversarial Detection Rate": results["adversarial_detection_rate"]
                })

        # Evaluate the combination of feature squeezers
        for attack_name, adversarial_attack in adversarial_attacks:
            print(f"Evaluating combined feature squeezers with {attack_name} attack")
            results = evaluate_with_feature_squeezing_grouped(
                model, 
                test_loader, 
                squeezer_combination["squeezers"], 
                squeezer_combination["threshold"],
                adversarial_attack=adversarial_attack
            )

            # Print results
            print(f"Combined Feature Squeezers Results with {attack_name}:")
            print("Legitimate Accuracy:", results["legitimate_accuracy"])
            print("Adversarial Detection Rate:", results["adversarial_detection_rate"])

            # Write to CSV
            writer.writerow({
                "Squeezer": "Combined (5-bit, Median 2x2, NLM 13-3-2)",
                "Adversarial Attack": attack_name,
                "Legitimate Accuracy": results["legitimate_accuracy"],
                "Adversarial Detection Rate": results["adversarial_detection_rate"]
            })

