import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, random_split


def simulation_files(directory, excluded=()):
    return [p for p in sorted(Path(directory).glob('sim_*.pt'))
            if re.fullmatch(r'sim_\d{4}\.pt', p.name) and int(p.stem[4:]) not in excluded]


def read_stats(directory):
    return json.loads((Path(directory) / 'global_stats.json').read_text())


class Lightcones(Dataset):
    def __init__(self, files, stats, crop_size=None):
        self.files = list(files)
        self.input_mean = float(stats['input_mean'])
        self.input_std = float(stats['input_std']) if stats['input_std'] > 1e-6 else 1.0
        self.target_mean = float(stats['target_mean'])
        self.target_std = max(float(stats['target_std']), 1e-6)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        blob = torch.load(self.files[index], map_location='cpu', weights_only=True)
        x, y = (blob['input'], blob['target']) if isinstance(blob, dict) else blob[:2]
        x, y = [v.float().reshape(1, *v.shape[-3:]) for v in (x, y)]
        if self.crop_size is not None:
            slices = tuple(slice((n-s)//2, (n-s)//2+s) for n, s in zip(x.shape[-3:], self.crop_size))
            if any(n < s for n, s in zip(x.shape[-3:], self.crop_size)):
                raise ValueError('Sample is smaller than crop_size.')
            x, y = x[(slice(None), *slices)], y[(slice(None), *slices)]
        return ((x-self.input_mean)/self.input_std).contiguous(), ((y-self.target_mean)/self.target_std).contiguous()


def split_dataset(dataset):
    nval = min(max(1, int(.1*len(dataset))), len(dataset)-1)
    if nval < 1:
        raise ValueError('At least two simulations are required.')
    return random_split(dataset, [len(dataset)-nval, nval], generator=torch.Generator().manual_seed(42))
