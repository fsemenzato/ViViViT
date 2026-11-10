import argparse
from pathlib import Path

import numpy as np
from scipy.signal.windows import tukey
import torch

from vivivit.power_spectrum import PkEstimator


def estimate(source, output, tag):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    estimator = PkEstimator(512., device)
    window = torch.tensor(tukey(128, 1.), dtype=torch.float32, device=device)[:, None, None]
    normalization = (window*window).mean().item()
    values = []
    for sid in range(1086, 1113):
        blob = torch.load(Path(source)/f'recon_sim_{sid:04d}.pt', map_location='cpu', weights_only=True)
        gt, rec = [blob[key].squeeze().float().to(device)*window for key in ['target', 'reconstruction']]
        result = estimator.compute(gt, rec, MAS=('CIC', 'CIC'))[0].cpu().numpy()
        result[:, 1:4] /= normalization
        values.append(result)
    values = np.stack(values)
    for index, key in enumerate(['k', 'gt', 'recon', 'cross']):
        np.save(output/f'Pk_{key}_{tag}.npy', values[..., index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['source', 'output', 'tag']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    estimate(args.source, args.output, args.tag)
