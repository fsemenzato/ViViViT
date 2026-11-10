import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def plot(directory, output):
    directory = Path(directory)
    blobs = [torch.load(directory/f'cml1_{key}_M256__recon_sim_0500.pt', map_location='cpu', weights_only=True) for key in ['vivit_causal', 'unet_tl_nl4', 'unet_tl_nl2']]
    fields = [blobs[0]['input'], blobs[0]['target'], *[b['reconstruction'] for b in blobs]]
    figure, axes = plt.subplots(1, 5, figsize=(12, 6))
    scale = float(blobs[0]['target'].float().abs().quantile(.99))
    for axis, field, name in zip(axes, fields, ['Input', 'Target', 'ViViViT', 'U-Net L=4', 'U-Net L=2']):
        axis.imshow(field.squeeze(0)[:, 32, :], aspect='auto', cmap='RdBu_r', vmin=-scale, vmax=scale)
        axis.set_title(name)
    axes[0].set_ylabel('Frame')
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
