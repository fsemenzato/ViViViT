import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .postprocess import load_bispectra, triangle_families, percentile_band


def plot_bispectrum(root, output, long_mode):
    root = Path(root)
    metadata = json.loads((root/'vivit/metadata.json').read_text())
    k = np.asarray(metadata['k_centers'])
    triangles = np.load(root/'vivit/tri_index_bin0.npy')
    families = triangle_families(triangles, k)
    names = ['squeezed', 'folded'] if long_mode else ['all']
    families['all'] = np.ones(len(triangles), dtype=bool)
    figure, axes = plt.subplots(len(names), 3, figsize=(12, 3*len(names)), sharex=False, sharey=True, squeeze=False)
    for depth in range(3):
        values = {model: load_bispectra(root/model, depth, long_mode) for model in ['vivit', 'unet']}
        for row, name in enumerate(names):
            selected = families[name]
            indices = np.flatnonzero(selected)
            indices = indices[np.lexsort((triangles[indices, 0], triangles[indices, 1], triangles[indices, 2]))]
            for model, label, color in [('vivit', 'ViViViT', '#0072B2'), ('unet', 'U-Net', '#D55E00')]:
                gt, rec = values[model]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.where(gt != 0, rec/gt, np.nan)[:, indices]
                lo, med, hi = percentile_band(ratio)
                axis = axes[row, 2-depth]
                axis.plot(np.arange(len(indices)), med, color=color, label=label)
                axis.fill_between(np.arange(len(indices)), lo, hi, color=color, alpha=.22)
                axis.axhline(1, color='black', ls='--', lw=.7)
                axis.set_ylim(.5, 2.)
            axes[row, 0].set_ylabel(name+'\n'+r'$B_{rec}/B_{GT}$')
    for axis in axes[-1]:
        axis.set_xlabel('Triangle index')
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--long-mode', action='store_true')
    args = parser.parse_args()
    plot_bispectrum(args.data, args.output, args.long_mode)


if __name__ == '__main__':
    main()
