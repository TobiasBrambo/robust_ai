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
        self.model = self.add_noise_layers(model, std_devs)

    def add_noise_layers(self, module, std_devs):
        """
        Recursively add noise layers before each Conv2d layer in the model.
        """
        init_noise, inner_noise = std_devs
        is_first_conv = True  # Track the first Conv2d layer

        def add_noise_recursive(submodule):
            nonlocal is_first_conv
            layers = []
            for name, layer in submodule.named_children():
                if isinstance(layer, nn.Conv2d):
                    # Add noise layer before the Conv2d layer
                    noise_std = init_noise if is_first_conv else inner_noise
                    layers.append(AddGaussianNoise(noise_std))
                    is_first_conv = False  # Set flag to False after first Conv2d layer
                    layers.append(layer)
                elif isinstance(layer, nn.Sequential) or len(list(layer.children())) > 0:
                    # Recurse if we have nested modules
                    layers.append(add_noise_recursive(layer))
                else:
                    # Otherwise, add the layer as is
                    layers.append(layer)
            return nn.Sequential(*layers)

        # Start the recursive process
        return add_noise_recursive(module)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    from torchvision.models import resnet18

    model = resnet18()
    print(model)
    wrapped_model = RSE(model)
    print(wrapped_model)

    input_tensor = torch.randn(1, 3, 224, 224)
    output = wrapped_model(input_tensor)
