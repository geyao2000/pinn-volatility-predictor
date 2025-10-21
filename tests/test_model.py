
import torch
from pinn_volatility.models.pinn import VolPINN
def test_forward():
    model = VolPINN(input_dim=5, hidden_dim=32, depth=2)
    x = torch.randn(2, 16, 5)
    y = model(x)
    assert y.shape == (2, 16)
