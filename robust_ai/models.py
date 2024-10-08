import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
from cpuinfo import get_cpu_info


if torch.cuda.is_available():
    device = torch.device('cuda')
    device_num = torch.cuda.current_device()
    device_info = torch.cuda.get_device_name(device_num)

else:
    device = torch.device('cpu')
    device_info = get_cpu_info()['brand_raw']

class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)  # adjust to match the flattened input
        self.fc2 = nn.Linear(128, 10)  # output 10 classes

    def forward(self, x):
        x = F.ReLU(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.ReLU(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)




class CNN(nn.Module):
    def __init__(self, n_classes=10):
        super(CNN, self).__init__()
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



    def fit(self, train_loader, test_loader, num_epochs=10, lr=0.01, loss_function=None, visualizer=None, disable_progress_bar=False, test_frequency=5):

        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        criterion = loss_function

        with tqdm.tqdm(total=num_epochs, disable=disable_progress_bar) as progress_bar:
            progress_bar.set_description(f"{device_info}: [N/A/{num_epochs}], Step [N/A/{len(train_loader)}], Loss: N/A, Test Acc: N/A")

            test_acc = 'N/A'

            results = {
                'best_acc': 0,
                'test_acc': []
            }

            for epoch in range(num_epochs):

                self.train()
                for i, (images, labels) in enumerate(train_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()

                    
                    if (i+1) % 100 == 0:
                        progress_bar.set_description(f"{device_info}: [{epoch+1}/{num_epochs}]: Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Acc: {test_acc}")
                    
                        if visualizer:
                            prob_dist = self.softmax(outputs)
                            visualizer.update_plots(prob_dist.cpu().detach().numpy(), labels.cpu().detach().numpy())

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


                progress_bar.set_description(f"{device_info}: [{epoch+1}/{num_epochs}]: Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Test Acc: {test_acc}") 
                progress_bar.update(1)

        return results
