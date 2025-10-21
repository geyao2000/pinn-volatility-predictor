
import argparse, pandas as pd
def main(ticker, start, end, out):
    try:
        import yfinance as yf
    except Exception:
        print('Install yfinance to fetch data: pip install yfinance'); return
    df = yf.download(ticker, start=start, end=end, progress=False)
    df = df.reset_index()[['Date','Adj Close']].rename(columns={'Date':'timestamp','Adj Close':'price'})
    df.to_csv(out, index=False); print(f'Saved {ticker} to {out}')
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ticker', default='SPY'); ap.add_argument('--start', default='2015-01-01')
    ap.add_argument('--end', default='2024-12-31'); ap.add_argument('--out', default='data/sample/spy.csv')
    args = ap.parse_args(); main(args.ticker, args.start, args.end, args.out)
