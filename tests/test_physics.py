
import torch
from pinn_volatility.physics.heston import heston_residual
def test_heston_residual_shapes():
    seq = torch.ones(100)*0.05
    res = heston_residual(seq, kappa=1.0, theta=0.05, sigma_v=0.25, dt=1.0)
    assert res.ndim == 1 and res.shape[0] == 99
