import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
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



def train_model(model, criterion, optimizer, epochs:int = 10):

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = fb.attacks.FGSM()

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
            
            # Generate adversarial examples using Foolbox
            adversarial_inputs = attack(fmodel, inputs, labels)
            
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




if __name__ == "__main__":

    # Load dataset (CIFAR-10 example)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    # Initialize model, loss, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, criterion, optimizer)
