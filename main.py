from pinn_volatility.train import train_main

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='configs/default.yaml')
    args = ap.parse_args(); train_main(args.config)
