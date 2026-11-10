import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vivivit.postprocess import power_ratios, percentile_band


def plot(data, output):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    for norm, color in [('L1', '#0072B2'), ('L2', '#D55E00')]:
        k, ratio, correlation = power_ratios(data, f'ablation_vivit_128cube_{norm}_1gpu')
        for axis, values in zip(axes, [ratio, correlation]):
            lo, med, hi = percentile_band(values[:, 0])
            axis.plot(k[0], med, color=color, label=norm)
            axis.fill_between(k[0], lo, hi, color=color, alpha=.22)
    for axis in axes:
        axis.set_xscale('log')
        axis.set_xlabel(r'$k\ [h/\mathrm{Mpc}]$')
        axis.axhline(1, color='black', ls='--', lw=.7)
    axes[0].set_ylabel(r'$P_{rec}/P_{GT}$')
    axes[1].set_ylabel(r'$r(k)$')
    axes[0].legend(frameon=False)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.data, args.output)
