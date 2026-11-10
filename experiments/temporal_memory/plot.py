import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(analysis, output):
    data = np.load(analysis)
    figure, axis = plt.subplots(figsize=(6, 4))
    m95 = [32, 64, 128, 256]
    for key, label, color in [('unet_tl_nl2', 'U-Net L=2', '#CC79A7'), ('unet_tl_nl3', 'U-Net L=3', '#E69F00'), ('unet_tl_nl4', 'U-Net L=4', '#D55E00'), ('vivit_causal', 'ViViViT', '#0072B2')]:
        values = [data[f'nmse_t__cml1_{key}_M{m}'][96:160].mean() for m in m95]
        axis.plot(m95, values, 'o-', color=color, label=label)
    axis.set(xlabel=r'$M_{95}$', ylabel='Interior NMSE', yscale='log')
    axis.legend(frameon=False)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plot(args.analysis, args.output)
