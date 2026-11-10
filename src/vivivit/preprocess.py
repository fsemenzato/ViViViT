import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F


def update_moments(state, value):
    count, mean, variance = state
    value = value.double()
    n, m, v = value.numel(), value.mean().item(), value.var(unbiased=False).item()
    if not count:
        return n, m, v
    total = count+n
    average = mean+(m-mean)*n/total
    return total, average, (count*(variance+mean*mean)+n*(v+m*m))/total-average*average


def read_field(path, key, start):
    if path.suffix == '.h5':
        with h5py.File(path) as file:
            array = file[key][...]
    else:
        array = np.load(path)
    array = np.moveaxis(array, -1, 0)[::-1].copy()
    array = array[start:start+384]
    if len(array) != 384:
        raise ValueError(f'Insufficient lightcone depth in {path}.')
    tensor = torch.from_numpy(array).float().unsqueeze(1)
    return F.interpolate(tensor, size=(128, 128), mode='bilinear', align_corners=False).squeeze(1).unsqueeze(0)


def preprocess(root, output, start, suffix):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs, targets = (0, 0., 0.), (0, 0., 0.)
    files = sorted((root/'temp').glob(f'temp_*{suffix}'), key=lambda p: int(p.stem.split('_')[-1]))
    if not files:
        raise FileNotFoundError(root/'temp')
    for path in files:
        sid = int(path.stem.split('_')[-1])
        destination = output/f'sim_{sid:04d}.pt'
        if destination.exists():
            raise FileExistsError(destination)
        x = read_field(path, 'temp', start)
        y = read_field(root/'dens'/path.name.replace('temp_', 'dens_', 1), 'dens', start)
        inputs, targets = update_moments(inputs, x), update_moments(targets, y)
        torch.save((x, y), destination)
    stats = dict(input_mean=inputs[1], input_std=inputs[2]**.5, target_mean=targets[1], target_std=targets[2]**.5)
    (output/'global_stats.json').write_text(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--start', type=int, default=300)
    parser.add_argument('--format', choices=['h5', 'npy'], default='h5')
    args = parser.parse_args()
    preprocess(args.root, args.output, args.start, '.'+args.format)


if __name__ == '__main__':
    main()
