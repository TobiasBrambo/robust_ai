from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class AddGaussianNoise(nn.Module):
    def __init__(self, std_dev):
        super(AddGaussianNoise, self).__init__()
        self.std_dev = std_dev

    def forward(self, x):
        noise = dist.Normal(0, self.std_dev).sample(x.size()).to(x.device)
        return x + noise



class RSE(nn.Module):
    def __init__(self, model, std_devs=(0.1, 0.05)):
        super(RSE, self).__init__()
        self.init_noise = std_devs[0]
        self.inner_noise = std_devs[1]
        self.did_first_noise = False
        self.model = self.add_noise_layers(model)

    def add_noise_layers(self, model):
        for child_name, child in model.named_children():
            if child_name == "downsample":
                continue
            if isinstance(child, nn.Conv2d):
                noise = self.init_noise if not self.did_first_noise else self.inner_noise
                setattr(model, child_name, nn.Sequential(OrderedDict([
                    (f"{'init_noise' if not self.did_first_noise else 'inner_noise'}", AddGaussianNoise(noise)),
                    ('conv', child)
                ])))
                self.did_first_noise = True
            else:
                self.add_noise_layers(child)

        return model

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from torchvision.models import resnext50_32x4d, resnet18

    model = resnet18()
    print(model)
    wrapped_model = RSE(model, std_devs=(0.1,0.1))
    print(wrapped_model)

    input_tensor = torch.randn(1, 3, 224, 224)
    output = wrapped_model(input_tensor)
    print(output.shape)
