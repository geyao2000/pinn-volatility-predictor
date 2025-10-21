
from __future__ import annotations
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from tqdm import tqdm
from .utils.config import load_config
from .utils.logging import ensure_dir, save_checkpoint
from .data.datasets import WindowedVolDataset
from .models.pinn import VolPINN
from .physics.heston import heston_residual, total_variation
def train_main(config_path: str = "configs/default.yaml"):
    cfg = load_config(config_path)
    torch.manual_seed(cfg['seed'])
    ds = WindowedVolDataset(cfg['data']['path'], cfg['data']['window'], cfg['data']['horizon'])
    n_train = int(len(ds) * cfg['data']['train_split'])
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg['train']['batch_size'])
    model = VolPINN(cfg['model']['input_dim'], cfg['model']['hidden_dim'], cfg['model']['depth']).to(cfg['device'])
    opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    kappa, theta, sigma_v, dt = cfg['physics']['kappa'], cfg['physics']['theta'], cfg['physics']['sigma_v'], cfg['physics']['dt']
    lambda_phys, lambda_tv = cfg['physics']['lambda_phys'], cfg['physics']['lambda_tv']
    global_step = 0
    for epoch in range(cfg['train']['epochs']):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['train']['epochs']}")
        for xb, yb in pbar:
            xb, yb = xb.to(cfg['device']), yb.to(cfg['device'])
            v_pred = model(xb)
            loss_data = F.mse_loss(v_pred, yb)
            phys = []; tvs = []
            for b in range(v_pred.shape[0]):
                seq = v_pred[b]
                res = heston_residual(seq, kappa, theta, sigma_v, dt)
                phys.append(torch.mean(res**2))
                tvs.append(total_variation(seq))
            loss_phys = torch.stack(phys).mean() if phys else torch.tensor(0.0, device=xb.device)
            loss_tv = torch.stack(tvs).mean() if tvs else torch.tensor(0.0, device=xb.device)
            loss = loss_data + lambda_phys*loss_phys + lambda_tv*loss_tv
            opt.zero_grad(); loss.backward(); opt.step()
            if global_step % cfg['train']['log_interval'] == 0:
                pbar.set_postfix(loss=float(loss.item()), data=float(loss_data.item()), phys=float(loss_phys.item()), tv=float(loss_tv.item()))
            global_step += 1
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg['device']), yb.to(cfg['device'])
                v_pred = model(xb)
                val_losses.append(F.mse_loss(v_pred, yb).item())
        print(f"Val MSE: {sum(val_losses)/max(1,len(val_losses)):.6f}")
    ckpt_path = ensure_dir(cfg['train']['ckpt_dir']) / cfg['train']['ckpt_name']
    save_checkpoint(model, str(ckpt_path))
    print(f"Saved checkpoint to {ckpt_path}")
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args(); train_main(args.config)
