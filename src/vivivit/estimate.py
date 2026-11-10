import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bispectrum import MuResolvedBispectrumGPU
from .power_spectrum import PkEstimator, ShellPkEstimator
from .taper import taper_field


def power_spectra(source, output, tag, ids, cube, device, taper=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if taper is None:
        estimator = PkEstimator(cube*4., device)
    else:
        estimator = ShellPkEstimator(cube*4., device)
    results = []
    for sid in ids:
        blob = torch.load(Path(source)/f'recon_sim_{sid:04d}.pt', map_location='cpu', weights_only=True)
        gt, rec = [blob[key][0].float() for key in ['target', 'reconstruction']]
        if gt.shape[0]%cube:
            raise ValueError('The LOS depth must be divisible by the cube size.')
        slabs = []
        for t in range(0, gt.shape[0], cube):
            if taper is None:
                slabs.append(estimator.compute(gt[t:t+cube], rec[t:t+cube], MAS=('CIC', 'CIC'))[0].cpu().numpy())
            else:
                (g, w2, _), (r, _, _) = taper_field(gt[t:t+cube].numpy(), taper), taper_field(rec[t:t+cube].numpy(), taper)
                values = estimator.compute(torch.tensor(g, dtype=torch.float32), torch.tensor(r, dtype=torch.float32))[0].cpu().numpy().astype(np.float64)
                values[:, 1:4] /= w2
                slabs.append(values)
        results.append(np.stack(slabs))
    results = np.stack(results)
    for column, name in enumerate(['k', 'gt', 'recon', 'cross']):
        np.save(output/f'Pk_{name}_{tag}.npy', results[..., column])
    (output/f'ids_{tag}.json').write_text(json.dumps(list(ids)))
    settings = dict(estimator='legacy', window='none', mas='CIC', binning='rounded shells, 63 bins') if taper is None else dict(estimator='tapered', window='tukey', alpha=taper, mas='none', binning='floor shells of width kF, independent-mode weights', normalization='divided by mean(w^2)')
    (output/f'settings_{tag}.json').write_text(json.dumps(dict(settings, box_cube=cube*4., cube=cube), indent=2))


def bispectra(source, output, ids, device, taper=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for depth in range(3):
        estimator = MuResolvedBispectrumGPU(512., 128, 5, device)
        arrays = {}
        for index, sid in enumerate(ids):
            blob = torch.load(Path(source)/f'recon_sim_{sid:04d}.pt', map_location='cpu', weights_only=True)
            for name, key in [('gt', 'target'), ('recon', 'reconstruction')]:
                full = blob[key].float().squeeze().numpy().astype(np.float64)
                full -= full.mean()
                cube = full[128*depth:128*(depth+1)].copy()
                cube -= cube.mean()
                if taper is None:
                    estimator.read_real(cube, MAS='CIC')
                    result = estimator.Bk_mu()
                else:
                    cube, w2, w3 = taper_field(cube, taper)
                    estimator.read_real(cube, MAS=None)
                    result = dict(estimator.Bk_mu())
                    result['B'] = result['B']/w3
                    result['P'] = result['P']/w2
                if not arrays:
                    shape = (len(ids), *result['B'].shape)
                    for field in ['gt', 'recon']:
                        arrays['B_'+field] = np.lib.format.open_memmap(output/f'B_{field}_bin{depth}.npy', mode='w+', dtype=np.float64, shape=shape)
                        arrays['P_'+field] = np.lib.format.open_memmap(output/f'P_{field}_bin{depth}.npy', mode='w+', dtype=np.float64, shape=(len(ids), *result['P'].shape))
                    for key in ['counts', 'tri_index']:
                        np.save(output/f'{key}_bin{depth}.npy', result[key])
                arrays['B_'+name][index] = result['B']
                arrays['P_'+name][index] = result['P']
        for array in arrays.values():
            array.flush()
        metadata = {key: result[key].tolist() for key in ['k_centers', 'mu_edges']}
        metadata.update(ids=list(ids), box_cube=512., grid=128, fc=1, dk=1, Nbins=42, mas='CIC' if taper is None else 'none', taper=taper)
        (output/'metadata.json').write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('statistic', choices=['power', 'bispectrum'])
    for name in ['source', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--tag', default='vivit_MLP')
    parser.add_argument('--ids', nargs=2, type=int, default=[0, 200])
    parser.add_argument('--cube', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--taper', type=float, default=None)
    args = parser.parse_args()
    if args.statistic == 'power':
        power_spectra(args.source, args.output, args.tag, range(*args.ids), args.cube, args.device, args.taper)
    else:
        bispectra(args.source, args.output, range(*args.ids), args.device, args.taper)


if __name__ == '__main__':
    main()
