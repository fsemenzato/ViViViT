from pathlib import Path

import numpy as np


def percentile_band(values):
    return np.nanpercentile(values, [16, 50, 84], axis=0)


def power_ratios(directory, tag):
    k, gt, rec, cross = [np.load(Path(directory)/f'Pk_{key}_{tag}.npy') for key in ['k', 'gt', 'recon', 'cross']]
    if k.ndim == 2:
        k, gt, rec, cross = [v[:, None, :] for v in [k, gt, rec, cross]]
    with np.errstate(divide='ignore', invalid='ignore'):
        return k.mean(axis=0)/.6727, rec/gt, cross/np.sqrt(gt*rec)


def collapse_orientations(bispectrum, counts, cells):
    output = []
    for index, value in enumerate(bispectrum):
        count = counts[index] if counts.ndim == bispectrum.ndim else counts
        numerator = (value*count*cells).sum(axis=(-3, -2, -1))
        denominator = (count*cells).sum(axis=(-3, -2, -1))
        with np.errstate(divide='ignore', invalid='ignore'):
            output.append(np.where(denominator > 0, numerator/denominator, np.nan))
    return np.stack(output)


def load_bispectra(directory, depth, long_mode=False):
    directory = Path(directory)
    gt, rec = [np.load(directory/f'B_{name}_bin{depth}.npy', mmap_mode='r') for name in ['gt', 'recon']]
    counts = np.load(directory/f'counts_bin{depth}.npy', mmap_mode='r')
    cells = np.ones(gt.shape[-3:], dtype=bool)
    if long_mode:
        cells[:4] = False
    return tuple(collapse_orientations(v, counts, cells) for v in [gt, rec])


def triangle_families(triangles, k):
    legs = k[triangles]
    smallest, middle, largest = legs.T
    rmin, rmid = smallest/largest, middle/largest
    equilateral = rmin >= .85
    squeezed = (rmin <= .5) & (rmid >= .8) & ~equilateral
    folded = (largest/(smallest+middle) >= .85) & (rmin > .5) & ~equilateral & ~squeezed
    return dict(squeezed=squeezed, folded=folded, equilateral=equilateral)

