import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .style import REDSHIFTS, paper_style


def plot_fields(vivit, unet, output, redshifts=REDSHIFTS, labels=('Input', 'Ground Truth', 'ViViViT', 'U-Net')):
    paper_style()
    a, b = [torch.load(p, map_location='cpu', weights_only=True) for p in [vivit, unet]]
    fields = [a['input'][0], a['target'][0], a['reconstruction'][0], b['reconstruction'][0]]
    middle = fields[2].shape[1]//2
    slices = [f[:, middle, :].float().numpy() for f in fields]
    figure = plt.figure(figsize=(9, 6))
    grid = figure.add_gridspec(1, 5, width_ratios=[1.0, 0.25, 1.0, 1.0, 1.0], wspace=0.05)
    axes = [figure.add_subplot(grid[0, i]) for i in [0, 2, 3, 4]]
    image_input = axes[0].imshow(slices[0][::-1], aspect='equal', origin='lower', vmin=slices[0].min(), vmax=slices[0].max(), cmap='inferno')
    rest = np.stack(slices[1:])
    image_rest = None
    for axis, field in zip(axes[1:], slices[1:]):
        image = axis.imshow(field[::-1], aspect='equal', origin='lower', vmin=rest.min(), vmax=rest.max(), cmap='coolwarm')
        image_rest = image_rest or image
    for axis, label in zip(axes, labels):
        axis.set_title(label, fontsize=15)
        axis.tick_params(which='both', direction='in', top=True, right=True)
        axis.tick_params(labelbottom=False, labeltop=False)
    z = np.loadtxt(redshifts)
    depth = slices[0].shape[0]
    ticks = np.linspace(0, depth-1, 6)
    names = [f'{z[depth-1-int(t)]:.2f}' for t in ticks]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels(names, fontsize=12)
    axes[0].set_ylabel('z', fontsize=15)
    axes[0].yaxis.set_label_position('left')
    axes[0].yaxis.tick_left()
    axes[0].tick_params(axis='y', labelleft=True, left=True, labelright=False, right=False)
    axes[3].set_yticks(ticks)
    axes[3].set_yticklabels(names, fontsize=12)
    axes[3].yaxis.set_label_position('right')
    axes[3].yaxis.tick_right()
    for axis in axes[1:]:
        axis.tick_params(axis='y', labelleft=False, labelright=False)
    plt.tight_layout()
    figure.canvas.draw()
    first, second, last = axes[0].get_position(), axes[1].get_position(), axes[3].get_position()
    height = 0.03
    bottom = min(axis.get_position().y0 for axis in axes)-0.04-height
    bar_input = figure.colorbar(image_input, cax=figure.add_axes([first.x0, bottom, first.width, height]), orientation='horizontal')
    bar_input.ax.set_xlabel(r'$\delta T_{21}$', fontsize=15)
    bar_input.ax.tick_params(labelsize=12)
    bar_rest = figure.colorbar(image_rest, cax=figure.add_axes([second.x0, bottom, last.x1-second.x0, height]), orientation='horizontal')
    bar_rest.ax.set_xlabel(r'$\delta_{\rm m}$', fontsize=15)
    bar_rest.ax.tick_params(labelsize=12)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches='tight')
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    for name in ['vivit', 'unet', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--redshifts', default=str(REDSHIFTS))
    args = parser.parse_args()
    plot_fields(args.vivit, args.unet, args.output, args.redshifts)


if __name__ == '__main__':
    main()
