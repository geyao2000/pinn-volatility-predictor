
import torch, pandas as pd, numpy as np
from .models.pinn import VolPINN
from .utils.config import load_config
def inference_main(ckpt:str, csv_path:str, config_path:str='configs/default.yaml'):
    cfg = load_config(config_path)
    df = pd.read_csv(csv_path)
    df['ret'] = np.log(df['price']).diff().fillna(0.0)
    rv = df['ret'].rolling(window=5).apply(lambda x: np.mean(x**2), raw=True).fillna(method='bfill')
    df['rv'] = rv
    t = np.arange(len(df))
    df['t_sin'] = np.sin(2*np.pi*t/252.0); df['t_cos'] = np.cos(2*np.pi*t/252.0); df['bias']=1.0
    X = df[['ret','rv','t_sin','t_cos','bias']].values.astype(np.float32)
    model = VolPINN(cfg['model']['input_dim'], cfg['model']['hidden_dim'], cfg['model']['depth'])
    state = torch.load(ckpt, map_location='cpu'); model.load_state_dict(state); model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).unsqueeze(0)
        v_pred = model(x).squeeze(0).numpy()
    out = pd.DataFrame({'timestamp': df['timestamp'], 'price': df['price'], 'v_pred': v_pred})
    out.to_csv('predictions.csv', index=False); print('Saved predictions to predictions.csv')
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--data', required=True); ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args(); inference_main(args.ckpt, args.data, args.config)
