
import torch
from torch.utils.data import DataLoader
import numpy as np
from .data.datasets import WindowedVolDataset
from .models.pinn import VolPINN
from .utils.config import load_config
from .utils.plots import plot_series
def evaluate_main(ckpt:str, data_path:str, config_path:str='configs/default.yaml'):
    cfg = load_config(config_path)
    ds = WindowedVolDataset(data_path, cfg['data']['window'], cfg['data']['horizon'])
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    model = VolPINN(cfg['model']['input_dim'], cfg['model']['hidden_dim'], cfg['model']['depth'])
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state); model.eval()
    preds = []; trues = []
    with torch.no_grad():
        for xb, yb in dl:
            v_pred = model(xb)
            preds.append(v_pred.squeeze(0).numpy())
            trues.append(yb.squeeze(0).numpy())
    preds = np.concatenate(preds); trues = np.concatenate(trues)
    plot_series(trues, preds, title='Volatility: true vs pred', savepath='figures_eval.png')
    print('Saved plot to figures_eval.png')
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--data', required=True); ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args(); evaluate_main(args.ckpt, args.data, args.config)
