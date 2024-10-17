import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import foolbox as fb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Example CNN model for image classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2)(x)
        x = x.view(-1, 32 * 8 * 8)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x



def train_model(model, criterion, optimizer, fmodel, attack, trainloader, epochs:int = 10):

    for epoch in range(epochs):  # Iterate through epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            

            _, adversarial_inputs, _= attack(fmodel, inputs, labels, epsilons=0.01)
            
            # Compute loss on adversarial inputs
            adv_outputs = model(adversarial_inputs)
            adv_loss = criterion(adv_outputs, labels)
            
            # Combine the loss (you can balance clean and adversarial loss)
            total_loss = loss + adv_loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += total_loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')

def evaluate(model, testloader):

    # correct = 0
    # total = 0
    #
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #
    #         outputs = model(images)
    #         _, pred = torch.max(outputs.data, 1)
    #
    #         total += labels.size(0)
    #         correct += (pred == labels).sum().item()

    clean_acc = fb.accuracy(fmodel, images, labels)
    


def retrain_resnet50(num_epochs:int = 1):

    batch_size = 64  # You can adjust this based on your available memory and resources
    learning_rate = 0.001

    # Load pre-trained ResNet-50 model
    model = models.resnet50(pretrained=True).to(device)

    # Update transformation for CIFAR-10 (32x32 images)
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR-10 images to 224x224 for ResNet-50
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        # Normalization for CIFAR-10 dataset
        transforms.Normalize([0.5, 0.5, 0.5], [0.5,0.5,0.5])
    ])

    # Create a foolbox model for adversarial attack generation
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1), preprocessing=None)
    attack = fb.attacks.LinfFastGradientAttack()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data/raw_files', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            # Move inputs and labels to the device (GPU or CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero out the gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass (clean inputs)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Generate adversarial examples
            _, adversarial_inputs, _ = attack(fmodel, inputs, labels, epsilons=0.01)
            adversarial_inputs = adversarial_inputs.to(device)

            # Forward pass (adversarial inputs)
            adv_outputs = model(adversarial_inputs)
            adv_loss = criterion(adv_outputs, labels)

            # Combine losses (adjust the ratio if needed)
            total_loss = loss + adv_loss

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

        # Print the loss for each epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}')

    # Save the trained model
    torch.save(model.state_dict(), './trained_models/resnet50_cifar10.pth')

    print(f'Finished Training, Loss: {loss.item():.4f}')


    
def evaluate_resnet50(model, fmodel, attack=None):


    model.to(device)
    model.eval()


    def accuracy(preds, labels):
        _, predicted = torch.max(preds, 1)
        return (predicted == labels).sum().item()

    
    for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            # Evaluate on regular images
            with torch.no_grad():
                outputs = model(images)
                regular_correct += accuracy(outputs, labels)

            # Evaluate on perturbed images (if attack is provided)
            if attack is not None:
                images_np = images.cpu().numpy()  # Convert to NumPy for Foolbox
                labels_np = labels.cpu().numpy()

                # Apply the adversarial attack
                perturbed_images_np = attack(fmodel, images_np, labels_np)
                perturbed_images = torch.tensor(perturbed_images_np).to(device)

                with torch.no_grad():
                    perturbed_outputs = model(perturbed_images)
                    perturbed_correct += accuracy(perturbed_outputs, labels)

        regular_accuracy = regular_correct / total
        perturbed_accuracy = perturbed_correct / total if attack is not None else None

        return {
            "regular_accuracy": regular_accuracy,
            "perturbed_accuracy": perturbed_accuracy
        }




if __name__ == "__main__":

    # Load dataset (CIFAR-10 example)
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = torchvision.datasets.CIFAR10(root='./data/raw_files', train=True, download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data/raw_files', train=False, download=True, transform=transform)
    # print(testset.shape())
    # testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    #
    #
    # assert False

    # Initialize model, loss, and optimizer
    # model = SimpleCNN()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    #
    # fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    # attack = fb.attacks.LinfFastGradientAttack()
    #
    #
    # train_model(model, criterion, optimizer, fmodel, attack, trainloader)
    #
    # torch.save(model.state_dict(), 'trained_models/')

    retrain_resnet50()
