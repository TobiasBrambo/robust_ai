'''ResNet in PyTorch with random self-ensemble.


ResNet implementation based on [1] taken from: https://github.com/kuangliu/pytorch-cifar
RSE implementation added based on [2], done by: https://github.com/TobiasBrambo


Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Xuanqing Liu, et. al.
    Towards Robust Neural Networks via Random self-ensemble arXiv:1712.00673v2
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, noise_std=0.1, skip_noise:bool = False):
        super(BasicBlock, self).__init__()

        self.noise_std = noise_std
        self.skip_noise = skip_noise

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def add_gaussian_noise(self, x):
        if not self.skip_noise:
            noise = dist.Normal(0, self.noise_std).sample(x.size()).to(x.device)
            return x + noise
        return x

    def forward(self, x):
        x = self.add_gaussian_noise(x)
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.add_gaussian_noise(out)
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, noise_std=0.1, skip_noise:bool = False):
        super(Bottleneck, self).__init__()

        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.skip_noise = skip_noise


        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def add_gaussian_noise(self, x):
        if not self.skip_noise:
            noise = dist.Normal(0, self.noise_std).sample(x.size()).to(x.device)
            return x + noise
        return x



    def forward(self, x):

        x = self.add_gaussian_noise(x)
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.add_gaussian_noise(out)
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.add_gaussian_noise(out)
        out = self.bn3(self.conv3(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, std_devs=(0.2, 0.1), num_classes=10):
        super(ResNet, self).__init__()
        self.init_noise = std_devs[0]
        self.inner_noise = std_devs[1]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()