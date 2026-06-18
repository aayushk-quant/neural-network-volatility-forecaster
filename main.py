import numpy as np
import os
import matplotlib.pyplot as plt
from src.data import download_prices, build_panel
from src.features import assemble
from src.evaluate import walk_forward, rmse, qlike, qlike_loss, diebold_mariano
os.makedirs('images', exist_ok=True)

def main():
    prices = download_prices('SPY', start = '2005-01-01', end = "2026-06-01")
    panel = build_panel(prices)
    X, Y = assemble(panel, horizons = (1, 5))
    returns = panel['ret']

    for H in (1,5):
        print(f"\n===== horizon: {H} day =====")
        res = walk_forward(X, Y, returns, H, initial_train=750, step=252).dropna()
        truth = res['truth']

        # scores
        print(f"{'model':8s} {'RMSE':>8s} {'QLIKE':>8s}")
        for m in ['naive', 'ewma', 'har', 'nn', 'garch']:
            print(f"{m:8s} {rmse(res[m], truth):8.4f} {qlike(res[m], truth):8.4f}")

        # significance: NN vs each baseline (negative DM = NN better)
        print("Diebold-Mariano (NN vs baseline):")
        for m in ['naive', 'ewma', 'har', 'garch']:
            dm, p = diebold_mariano(qlike_loss(res['nn'], truth), qlike_loss(res[m], truth), H)
            print(f"  nn vs {m:6s}: DM={dm:+.2f}  p={p:.3f}")

        # plot (exp back to vol for readability) for the writeup
        np.exp(res.loc['2019':'2021', ['truth', 'har', 'nn']]).plot(figsize=(11, 4), lw=0.8,
            title=f'{H}-day annualised vol (2019-2021)')
        plt.ylabel('annualised vol'); plt.tight_layout()
        plt.savefig(f'images/forecast_h{H}.png', dpi=120); plt.close()
        print(f"saved forecast_h{H}.png")

if __name__ == '__main__':
    main()