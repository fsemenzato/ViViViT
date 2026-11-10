import argparse
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np


def generate(config, output, cache, ids, seed):
    import py21cmfast as p21c
    output = Path(output)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for sid in ids:
        paths = {key: output/key/f'{key}_{sid:04d}.h5' for key in ['temp', 'dens']}
        if any(p.exists() for p in paths.values()):
            raise FileExistsError(f'Simulation {sid} already exists.')
        astro = {key: float(rng.uniform(*bounds)) for key, bounds in config['astro_ranges'].items()}
        random_seed = int(rng.integers(1, 2**31))
        initial = p21c.initial_conditions(user_params=config['user_params'], cosmo_params=p21c.CosmoParams(**config['cosmo_params']), random_seed=random_seed, direc=str(cache))
        lightcone = p21c.run_lightcone(redshift=config['redshift'], max_redshift=config['max_redshift'], init_box=initial, flag_options=config['flag_options'], astro_params=p21c.AstroParams(astro), lightcone_quantities=['brightness_temp', 'density'], direc=str(cache))
        for key, attribute in [('temp', 'brightness_temp'), ('dens', 'density')]:
            paths[key].parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(paths[key], 'w') as file:
                file.create_dataset(key, data=getattr(lightcone, attribute))
                file.create_dataset('redshifts', data=lightcone.lightcone_redshifts)
                file.attrs['parameters'] = json.dumps(dict(config=config, astro_params=astro, random_seed=random_seed, py21cmfast=importlib.metadata.version('21cmFAST')))


def main():
    parser = argparse.ArgumentParser()
    for name in ['config', 'output', 'cache']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--ids', type=int, nargs=2, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    generate(json.loads(Path(args.config).read_text()), args.output, args.cache, range(*args.ids), args.seed)


if __name__ == '__main__':
    main()
