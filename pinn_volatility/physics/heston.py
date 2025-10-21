
import torch
import torch.nn.functional as F
def softplus_clamp(x, beta:float=10.0, threshold:float=20.0):
    return F.softplus(x, beta=beta, threshold=threshold)
def heston_residual(v_pred, kappa: float, theta: float, sigma_v: float, dt: float):
    eps = 1e-6
    v = softplus_clamp(v_pred)
    dv = v[1:] - v[:-1]
    drift = kappa*(theta - v[:-1])*dt
    scale = torch.sqrt(torch.clamp(v[:-1]*dt, min=eps))
    res = (dv - drift) / (scale + eps)
    return res
def total_variation(v_pred):
    return torch.mean(torch.abs(v_pred[1:] - v_pred[:-1]))
