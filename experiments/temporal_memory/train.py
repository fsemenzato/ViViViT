import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vivivit.data import Lightcones, simulation_files, read_stats, split_dataset
from vivivit.loss import reconstruction_loss
from vivivit.temporal_models import build_model


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()


def train(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(config['seed'])
    data = Path(config['preproc_video_dir'])
    config.update(read_stats(data))
    manifest = json.loads((data/'manifest.json').read_text())
    excluded = {int(k) for k, v in manifest.items() if v['mode'].startswith('eval')}
    train_set, validation_set = split_dataset(Lightcones(simulation_files(data, excluded), config))
    training = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    validation = DataLoader(validation_set, batch_size=config['batch_size'])
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    accumulation = config['grad_accum_steps']
    total = config['num_epochs']*max(1, math.ceil(len(training)/accumulation))
    warmup = config['warmup_steps']
    def factor(step):
        if step < warmup:
            return step/warmup
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    out = Path(config['output_base'])/config['run_name']
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    best, patience = float('inf'), 0
    for epoch in range(config['num_epochs']):
        model.train()
        optimizer.zero_grad()
        for step, (x, y) in enumerate(training):
            with autocast(device):
                loss = reconstruction_loss(model(x.to(device)), y.to(device))/accumulation
            loss.backward()
            if (step+1)%accumulation == 0 or step+1 == len(training):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        model.eval()
        total_loss, total_mse, count = 0., 0., 0
        with torch.no_grad():
            for x, y in validation:
                y = y.to(device)
                with autocast(device):
                    prediction = model(x.to(device))
                    loss = reconstruction_loss(prediction, y)
                    mse = torch.nn.functional.mse_loss(prediction, y)
                total_loss += float(loss)*len(x)
                total_mse += float(mse)*len(x)
                count += len(x)
        value = total_loss/count
        print(f'Epoch {epoch+1}: validation={value:.8g}', flush=True)
        if value < best:
            best, patience = value, 0
            torch.save(dict(model=model.state_dict(), config=config, epoch=epoch+1, val_loss=value, val_mse=total_mse/count), out/'best_model.pth')
        else:
            patience += 1
        if patience >= config['early_stopping_patience']:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()))
