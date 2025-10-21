
import numpy as np, pandas as pd, argparse
def synth_heston(n_steps=1500, s0=100.0, v0=0.04, mu=0.0, kappa=1.5, theta=0.04, sigma_v=0.25, rho=-0.6, dt=1.0/252):
    rng = np.random.default_rng(42)
    S = np.zeros(n_steps); V = np.zeros(n_steps); S[0]=s0; V[0]=v0
    for t in range(1, n_steps):
        z1 = rng.normal(); z2 = rng.normal()
        w1 = z1; w2 = rho*z1 + np.sqrt(1-rho**2)*z2
        V[t] = np.maximum(V[t-1] + kappa*(theta - V[t-1])*dt + sigma_v*np.sqrt(max(V[t-1],1e-8))*np.sqrt(dt)*w2, 1e-8)
        S[t] = S[t-1]*np.exp((mu - 0.5*V[t-1])*dt + np.sqrt(V[t-1]*dt)*w1)
    return S, V
def main(out_path, n_steps):
    prices, vol = synth_heston(n_steps=n_steps)
    ts = pd.date_range('2020-01-01', periods=n_steps, freq='B')
    df = pd.DataFrame({'timestamp': ts.astype(str), 'price': prices, 'true_vol': vol})
    df.to_csv(out_path, index=False); print(f'Saved synthetic data to {out_path}')
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/sample/synth_prices.csv')
    ap.add_argument('--n_steps', type=int, default=1200)
    args = ap.parse_args(); main(args.out, args.n_steps)
