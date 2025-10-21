# Physics-Informed Neural Networks for Predicting Market Volatility

A research-grade repository applying **PINNs** to forecast market volatility by embedding **stochastic volatility dynamics** (Heston/OU) into the training loss.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_synthetic.py --out data/sample/synth_prices.csv --n_steps 1000
python main.py --config configs/default.yaml
python -m pinn_volatility.evaluate --ckpt runs/last.pt --data data/sample/synth_prices.csv
```
