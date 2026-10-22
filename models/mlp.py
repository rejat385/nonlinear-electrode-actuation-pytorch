import torch
from torch import nn

class ElectrodeMLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 7)
        )

    def forward (self, x):
        return self.network(x)


if __name__ == "__main__":
    model = ElectrodeMLP()
    sample = torch.rand(1, 7)
    print(model(sample).shape)