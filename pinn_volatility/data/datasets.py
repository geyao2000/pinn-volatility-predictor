
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
class WindowedVolDataset(Dataset):
    def __init__(self, csv_path:str, window:int=32, horizon:int=1):
        df = pd.read_csv(csv_path)
        df['ret'] = np.log(df['price']).diff().fillna(0.0)
        rv = df['ret'].rolling(window=5).apply(lambda x: np.mean(x**2), raw=True).fillna(method='bfill')
        df['rv'] = rv
        t = np.arange(len(df))
        df['t_sin'] = np.sin(2*np.pi*t/252.0)
        df['t_cos'] = np.cos(2*np.pi*t/252.0)
        df['bias']  = 1.0
        self.X = df[['ret','rv','t_sin','t_cos','bias']].values.astype(np.float32)
        self.y = df['rv'].values.astype(np.float32)
        self.window = window; self.horizon = horizon
    def __len__(self):
        return len(self.X) - self.window - self.horizon + 1
    def __getitem__(self, idx):
        x = self.X[idx:idx+self.window]
        y = self.y[idx:idx+self.window]
        return torch.from_numpy(x), torch.from_numpy(y)
