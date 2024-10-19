import torch
import torch.nn as nn
import torch.optim as optim

import tqdm
from time import perf_counter
from cpuinfo import get_cpu_info

import foolbox as fb
# alternativ: https://github.com/ndb796/Pytorch-Adversarial-Training-CIFAR

from data import cifar10


if torch.cuda.is_available():
    device = torch.device('cuda')
    device_num = torch.cuda.current_device()
    device_info = torch.cuda.get_device_name(device_num)

else:
    device = torch.device('cpu')
    device_info = get_cpu_info()['brand_raw']

def has_converged(accuracies, tolerance=1.0, patience=5):
    """Checks if the model has converged based on the change in accuracy."""
    
    if len(accuracies) < patience:
        return False
    # Check if the improvement in accuracy is less than tolerance for last `patience` epochs
    for i in range(1, patience + 1):
        if abs(accuracies[-i] - accuracies[-i - 1]) > tolerance:
            return False
    return True


class CNN(nn.Module):
    def __init__(self, n_classes=10, converger=None):
        super(CNN, self).__init__()

        self.converger = converger

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(128 * 4 * 4, 1000)
        self.fc2 = nn.Linear(1000, n_classes)  
        
        self.dropout = nn.Dropout(p=0.1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        
        return x

    def save_model(self, save_name:str = "CNN_model.pth"):

        save_path = f"trained_models/{save_name}"
        torch.save(self.state_dict(), save_path)

        print(f"Saved model to: {save_path}")


    def fit(self, train_loader, test_loader, num_epochs=10, lr=0.001, loss_function=None, visualizer=None, disable_progress_bar=False, test_frequency=5):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = loss_function

        with tqdm.tqdm(total=num_epochs, disable=disable_progress_bar) as progress_bar:
            progress_bar.set_description(f"{device_info}: [N/A/{num_epochs}], Step [N/A/{len(train_loader)}], Loss: N/A, Test Acc: N/A")

            test_acc = 'N/A'
            have_converged = False
            results = {
                'best_acc': 0,
                'test_acc': [],
                'train_time' : []
            }

            for epoch in range(num_epochs):

                st = perf_counter()
                self.train()
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    
                    if (i+1) % 100 == 0:
                        progress_bar.set_description(f"{device_info}: [{epoch+1}/{num_epochs}]: Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Acc: {test_acc}")
                    

                et = perf_counter()
                
                if (epoch+1) % test_frequency == 0:

                    self.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = self(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                        test_acc = 100 * correct / total

                        results['test_acc'].append(test_acc)
                        results['best_acc'] = max(results['best_acc'], test_acc)

                results['train_time'].append(et-st)

                if self.converger is not None and not have_converged:

                    if self.converger(results['test_acc']):
                        
                        have_converged = True
                        results['converged_at_epoch'] = epoch

                progress_bar.set_description(f"{device_info}: [{epoch+1}/{num_epochs}]: Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Acc: {test_acc}") 
                progress_bar.update(1)

        return results


    def evaluate(self, test_loader):

        self.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            true_labels = []
            pred_labels = []

            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                true_labels.append(labels)
                pred_labels.append(predicted)

            test_acc = 100 * correct / total

        return test_acc, true_labels, pred_labels



    def fit_with_adversaries(self, train_loader, test_loader, num_epochs=10, lr=0.001, loss_function=None, disable_progress_bar=False, test_frequency=5):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = loss_function

        with tqdm.tqdm(total=num_epochs, disable=disable_progress_bar) as progress_bar:
            progress_bar.set_description(f"{device_info}: [N/A/{num_epochs}], Step [N/A/{len(train_loader)}], Loss: N/A, Test Acc: N/A")

            test_acc = 'N/A'
            have_converged = False
            results = {
                'best_acc': 0,
                'test_acc': [],
                'train_time' : []
            }

            for epoch in range(num_epochs):

                st = perf_counter()
                self.train()
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    
                    if (i+1) % 100 == 0:
                        progress_bar.set_description(f"{device_info}: [{epoch+1}/{num_epochs}]: Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Acc: {test_acc}")
                    

                et = perf_counter()
                
                if (epoch+1) % test_frequency == 0:

                    self.eval()
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for images, labels in test_loader:
                            images = images.to(device)
                            labels = labels.to(device)
                            outputs = self(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                        test_acc = 100 * correct / total

                        results['test_acc'].append(test_acc)
                        results['best_acc'] = max(results['best_acc'], test_acc)

                results['train_time'].append(et-st)

                if self.converger is not None and not have_converged:

                    if self.converger(results['test_acc']):
                        
                        have_converged = True
                        results['converged_at_epoch'] = epoch

                progress_bar.set_description(f"{device_info}: [{epoch+1}/{num_epochs}]: Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Acc: {test_acc}") 
                progress_bar.update(1)

        return results




def train_cnn_base(num_epochs:int = 50):
    train_loader, test_loader, classes = cifar10.get_cifar10(n_classes=0, batch_size=128, seed=42)

    loss_func = nn.CrossEntropyLoss()
    model = CNN(converger=has_converged).to(device)

    results = model.fit(train_loader, test_loader, loss_function=loss_func, num_epochs=num_epochs, test_frequency=5)

    model.save_model(save_name=f"CNN_base_pretrained_cifar10_{num_epochs}ep.pth")

    print(results)



if __name__ == "__main__":
    train_cnn_base()
