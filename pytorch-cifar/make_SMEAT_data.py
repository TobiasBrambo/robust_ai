import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from advertorch.attacks import LinfPGDAttack, GradientSignAttack, L1PGDAttack, L2PGDAttack, DeepfoolLinfAttack

import torchvision.utils as vutils
import os
from models import *  # Import your models

def make_data(model, attack, model_name, attack_name, batch_size=64):
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    device = "cuda"
    
    output_dir = f'./data/perturbed_train/{model_name}/{attack_name}'
    os.makedirs(output_dir, exist_ok=True)

    img_index = 0  # To keep track of global image numbering
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        # Generate adversarial images for the batch
        adv_images = attack.perturb(images, labels)

        # save images
        for j in range(images.size(0)):
            label = labels[j].item()
            # Save image with label in filename
            img_index = i * batch_size + j  # Correct global index for each image
            image_path = os.path.join(output_dir, f"img_{img_index}_label_{label}.png")
            vutils.save_image(adv_images[j], image_path)
        
        if i % 10 == 0:
            print(f"[{model_name} - {attack_name}] Processed {i}/{len(trainloader)} batches.")





if __name__ == "__main__":
    device = "cuda"

    # Define models
    model_dict = {
        "resnet50": ResNet50(),
        # "resnet18": ResNet18(),
        # "SimpleDLA": SimpleDLA(),
        # "DenseNet121": DenseNet121(),
        # "LeNet": LeNet(),
    }

    # Define attacks with parameters included in names
    attack_dict = {
        "resnet50": [
            ("LinfPGD_eps005_iter10_step009", lambda model: LinfPGDAttack(model, nn.CrossEntropyLoss(), eps=0.05, nb_iter=10, eps_iter=0.009)),
            ("L1PGD_eps005_iter10_step008", lambda model: L1PGDAttack(model, nn.CrossEntropyLoss(), eps=0.05, nb_iter=10, eps_iter=0.008)),
            ("FGSM_eps03", lambda model: GradientSignAttack(model, nn.CrossEntropyLoss(), eps=0.3)),
            ("DeepFool_eps005_iter10", lambda model: DeepfoolLinfAttack(model, num_classes=10, nb_iter=10, eps=0.05, loss_fn=nn.CrossEntropyLoss())),
        ],
        # "LeNet": [
        #     ("LinfPGD_eps005_iter10_step01", lambda model: LinfPGDAttack(model, nn.CrossEntropyLoss(), eps=0.05, nb_iter=10, eps_iter=0.01)),
        #     ("L2PGD_eps005_iter10_step01", lambda model: L2PGDAttack(model, nn.CrossEntropyLoss(), eps=0.05, nb_iter=10, eps_iter=0.01)),
        # ],
        # "SimpleDLA": [
        #     # ("FGSM_default_params", lambda model: GradientSignAttack(model, nn.CrossEntropyLoss())),
        #     ("DeepFool_eps005_iter10", lambda model: DeepfoolLinfAttack(model, num_classes=10, nb_iter=10, eps=0.05, loss_fn=nn.CrossEntropyLoss())),
        # ],
        # "DenseNet121": [
        #     ("LinfPGD_eps005_iter10_step01", lambda model: LinfPGDAttack(model, nn.CrossEntropyLoss(), eps=0.05, nb_iter=10, eps_iter=0.01)),
        #     ("L2PGD_eps005_iter10_step01", lambda model: L2PGDAttack(model, nn.CrossEntropyLoss(), eps=0.05, nb_iter=10, eps_iter=0.01)),
        # ],
    }

    for model_name, model_class in model_dict.items():
        print(f"Running for model: {model_name}")

        # Initialize and load the model
        model = model_class.to(device)
        model = torch.nn.DataParallel(model)

        # Load pre-trained weights
        checkpoint_path = f'./checkpoint/{model_name}_clean/best_regular_model.pth'
        if os.path.exists(checkpoint_path):
            model_check = torch.load(checkpoint_path)
            model.load_state_dict(model_check)
        else:
            print(f"Warning: Checkpoint not found for {model_name}, skipping.")
            continue

        if model_name in attack_dict:
            for attack_name, attack_init in attack_dict[model_name]:
                print(f"Running attack: {attack_name}")

                # Initialize the attack
                attack = attack_init(model)

                # Generate data
                make_data(model, attack, model_name, attack_name, batch_size=64)
        else:
            print(f"No attacks defined for model: {model_name}, skipping.")
