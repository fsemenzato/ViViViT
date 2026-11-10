import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .postprocess import power_ratios
from .style import REDSHIFTS, paper_style


def slab_titles(redshifts, slabs, cube=128):
    titles = []
    for slab in slabs:
        z = np.sort(redshifts[cube*slab:cube*(slab+1)])
        titles.append(f"z $\\in$ [{z[0]:.2f}, {z[-1]:.2f}]")
    return titles


def plot_power(directory, output, tags, labels=('ViViViT', 'U-Net'), redshifts=REDSHIFTS, box_cube=512., cube=128):
    paper_style()
    nyquist = np.pi*cube/box_cube
    colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    figure, ax = plt.subplots(2, 3, figsize=(12, 4.5), sharex='col', sharey='row')
    plt.subplots_adjust(hspace=0, wspace=0, bottom=0.14)
    titles = slab_titles(np.loadtxt(redshifts), [2, 1, 0], cube)
    for tag, label, color in zip(tags, labels, [colors[0], colors[1]]):
        k, ratio, correlation = power_ratios(directory, tag)
        keep = k[0]*.6727 <= nyquist*(1+1e-6)
        ks = k[0][keep]
        bands = []
        for values in [ratio, correlation]:
            values = values[..., keep]
            bands.append([np.median(values, axis=0)[::-1]]+[np.percentile(values, q, axis=0)[::-1] for q in [16, 84, 2.5, 97.5]])
        for column in range(3):
            ax[0, column].set_title(titles[column], fontsize=15)
            for row, (median, lo, hi, llo, hhi) in enumerate(bands):
                kwargs = dict(label=label if column == 0 else '_nolegend_') if row == 0 else {}
                ax[row, column].plot(ks, median[column], color=color, rasterized=True, **kwargs)
                ax[row, column].fill_between(ks, lo[column], hi[column], alpha=0.3, color=color, rasterized=True)
                ax[row, column].fill_between(ks, llo[column], hhi[column], alpha=0.1, color=color, rasterized=True)
                ax[row, column].axhline(1.0, color='black', linestyle='--')
            ax[0, column].tick_params(labelleft=True, labelsize=12)
    for axis in ax[-1, :]:
        axis.set_xlabel(r'$k\,[h/\mathrm{Mpc}]$', fontsize=15)
        axis.set_xlim(ks[0], ks[-1])
        axis.tick_params(labelbottom=True, labelsize=12)
    ax[0, 0].set_ylabel(r'$P_{\rm rec} / P_{\rm GT}$', fontsize=15)
    ax[1, 0].set_ylabel(r'$r$', fontsize=15)
    ax[0, 0].set_ylim(0.9, 1.05)
    ax[1, 0].set_ylim(0.96, 1.015)
    for axis in ax.flat:
        axis.tick_params(which='both', direction='in', top=True, right=True)
    for row in range(2):
        for column in range(1, 3):
            ax[row, column].tick_params(labelleft=False)
    figure.canvas.draw()
    ticks = ax[0, 0].get_yticklabels()
    if ticks:
        ticks[0].set_visible(False)
    ax[0, 0].tick_params(labelsize=12)
    handles, names = ax[0, 0].get_legend_handles_labels()
    figure.legend(handles, names, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=15, ncol=len(names), frameon=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tags', nargs=2, default=['vivit_MLP', 'unet'])
    parser.add_argument('--labels', nargs=2, default=['ViViViT', 'U-Net'])
    parser.add_argument('--redshifts', default=str(REDSHIFTS))
    args = parser.parse_args()
    plot_power(args.data, args.output, args.tags, args.labels, args.redshifts)


if __name__ == '__main__':
    main()
