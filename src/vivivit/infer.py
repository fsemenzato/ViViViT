import argparse
import json
from pathlib import Path

import torch

from .data import Lightcones, read_stats
from .models import build_model


@torch.no_grad()
def infer(checkpoint, data, output, ids, device, model_factory=build_model):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = {**saved['config'], **read_stats(data)}
    model = model_factory(config).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    dataset = Lightcones([Path(data)/f'sim_{i:04d}.pt' for i in ids], config, config.get('crop_size'))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    for i, (x, y) in zip(ids, dataset):
        prediction = model(x.unsqueeze(0).to(device))[0].cpu()
        torch.save(dict(input=x*dataset.input_std+dataset.input_mean,
                        target=y*dataset.target_std+dataset.target_mean,
                        reconstruction=prediction*dataset.target_std+dataset.target_mean), out/f'recon_sim_{i:04d}.pt')
    (out/'inference.json').write_text(json.dumps(dict(checkpoint=str(Path(checkpoint).resolve()), data=str(Path(data).resolve()), ids=list(ids), normalization=read_stats(data)), indent=2))


def main():
    parser = argparse.ArgumentParser()
    for arg in ['checkpoint', 'data', 'output']:
        parser.add_argument('--'+arg, required=True)
    parser.add_argument('--ids', type=int, nargs=2, default=[0, 200])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    infer(args.checkpoint, args.data, args.output, range(*args.ids), args.device)


if __name__ == '__main__':
    main()
