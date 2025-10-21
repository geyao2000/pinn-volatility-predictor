
import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, depth:int, output_dim:int=1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, output_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
class VolPINN(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, depth=4):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, depth, 1)
    def forward(self, x):  # x: [B, T, F]
        B, T, F = x.shape
        v = self.mlp(x.reshape(B*T, F)).reshape(B, T, 1).squeeze(-1)
        return v
