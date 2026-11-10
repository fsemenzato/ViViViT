import argparse
import json
from pathlib import Path

import torch

from vivivit.data import Lightcones, read_stats
from vivivit.temporal_models import build_model


@torch.no_grad()
def infer(checkpoint, data, output):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = {**saved['config'], **read_stats(data)}
    manifest = json.loads((Path(data)/'manifest.json').read_text())
    ids = sorted(int(k) for k, v in manifest.items() if v['mode'].startswith('eval'))
    dataset = Lightcones([Path(data)/f'sim_{i:04d}.pt' for i in ids], config)
    model = build_model(config).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for sid, (x, y) in zip(ids, dataset):
        prediction = model(x.unsqueeze(0).to(device))[0].cpu()
        torch.save(dict(input=(x*dataset.input_std+dataset.input_mean).half(), target=(y*dataset.target_std+dataset.target_mean).half(), reconstruction=(prediction*dataset.target_std+dataset.target_mean).half(), params=manifest[str(sid)]), output/f'recon_sim_{sid:04d}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['checkpoint', 'data', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    infer(args.checkpoint, args.data, args.output)
