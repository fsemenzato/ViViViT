import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vivivit.temporal_data import a_of_M95, sample_forcing, coupled_ou


def generate(output, m95, ids, seed, mode):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest_path = output/'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for sid in ids:
        path = output/f'sim_{sid:04d}.pt'
        if path.exists():
            raise FileExistsError(path)
        sim_seed = int(rng.integers(0, 2**31))
        a = a_of_M95(m95)
        x, y = coupled_ou(sample_forcing(256, 64, np.random.default_rng(sim_seed)), a)
        torch.save((torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)), path)
        manifest[str(sid)] = dict(M95=m95, a=a, variant='L0', seed=sim_seed, mode=mode)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if mode == 'train':
        (output/'global_stats.json').write_text(json.dumps(dict(input_mean=0., input_std=1., target_mean=0., target_std=1.), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--M95', type=int, choices=[32, 64, 128, 256], required=True)
    parser.add_argument('--ids', type=int, nargs=2, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--mode', choices=['train', 'eval'], required=True)
    args = parser.parse_args()
    generate(args.output, args.M95, range(*args.ids), args.seed, args.mode)
