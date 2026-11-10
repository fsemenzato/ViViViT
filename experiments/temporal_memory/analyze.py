import argparse
from pathlib import Path

import numpy as np
import torch


def analyze(root, output):
    root = Path(root)
    result, runs = {}, []
    for run in sorted(root.glob('cml1_*_M*')):
        if not any(run.name.startswith('cml1_'+k+'_M') for k in ['unet_tl_nl2', 'unet_tl_nl3', 'unet_tl_nl4', 'vivit_causal']):
            continue
        numerators, denominators = [], []
        for path in sorted((run/'recon').glob('recon_sim_*.pt')):
            blob = torch.load(path, map_location='cpu', weights_only=True)
            target, prediction = [blob[key].double().squeeze(0).numpy() for key in ['target', 'reconstruction']]
            numerators.append(((prediction-target)**2).sum(axis=(1, 2)))
            denominators.append((target**2).sum(axis=(1, 2)))
        if not numerators:
            raise FileNotFoundError(run/'recon')
        numerators, denominators = np.asarray(numerators), np.asarray(denominators)
        result['nmse_t__'+run.name] = numerators.sum(axis=0)/denominators.sum(axis=0)
        result['numer_sim__'+run.name], result['denom_sim__'+run.name] = numerators, denominators
        runs.append(run.name)
    if not runs:
        raise FileNotFoundError(root)
    np.savez(output, runs=runs, **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(args.runs, args.output)
