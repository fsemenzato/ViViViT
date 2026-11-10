import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from .data import Lightcones, read_stats, simulation_files, split_dataset
from .loss import reconstruction_loss
from .models import build_model


def warmup_cosine(optimizer, warmup_steps, total_steps):
    def factor(step):
        if step < warmup_steps:
            return step/max(1, warmup_steps)
        progress = (step-warmup_steps)/max(1, total_steps-warmup_steps)
        return .5*(1+torch.cos(torch.tensor(progress*3.1415926535))).item()
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device):
    return torch.autocast('cuda') if str(device).startswith('cuda') else nullcontext()


def train(config, resume=None, model_factory=build_model):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    distributed = local_rank >= 0
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl')
    rank = dist.get_rank() if distributed else 0
    device = f'cuda:{local_rank}' if distributed else ('cuda' if torch.cuda.is_available() else 'cpu')
    config = {**config, **read_stats(config['preproc_video_dir'])}
    out = Path(config['output_base'])/config['run_name']
    out.mkdir(parents=True, exist_ok=True)
    files = simulation_files(config['preproc_video_dir'], config['test_indices'])
    if 'max_trainval_samples' in config:
        indices = torch.randperm(len(files), generator=torch.Generator().manual_seed(42))[:config['max_trainval_samples']]
        files = [files[i] for i in indices]
    dataset = Lightcones(files, config, config.get('crop_size'))
    train_set, valid_set = split_dataset(dataset)
    if rank == 0:
        (out/'config.json').write_text(json.dumps(config, indent=2))
        (out/'split.json').write_text(json.dumps({name: [str(files[i]) for i in subset.indices] for name, subset in [('train', train_set), ('validation', valid_set)]}, indent=2))
    train_sampler = DistributedSampler(train_set) if distributed else None
    valid_sampler = DistributedSampler(valid_set, shuffle=False) if distributed else None
    loader_args = dict(batch_size=config['batch_size'], num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=1)
    train_loader = DataLoader(train_set, shuffle=train_sampler is None, sampler=train_sampler, **loader_args)
    valid_loader = DataLoader(valid_set, sampler=valid_sampler, **loader_args)
    model = model_factory(config).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'], betas=(.9, .95))
    scaler = torch.cuda.amp.GradScaler(enabled=str(device).startswith('cuda'))
    scheduler = warmup_cosine(optimizer, config['warmup_steps'], config['num_epochs']*len(train_loader))
    start, best, patience = 0, float('inf'), 0
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)
        (model.module if distributed else model).load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optim'])
        scheduler.load_state_dict(saved['sched'])
        start, best = saved['epoch'], saved['best']
    accumulation = config['grad_accum_steps']
    for epoch in range(start, config['num_epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader):
            update = (step+1)%accumulation == 0 or step+1 == len(train_loader)
            context = model.no_sync() if distributed and not update else nullcontext()
            with context:
                with autocast(device):
                    loss = reconstruction_loss(model(x.to(device)), y.to(device), config.get('loss_norm', 'L1'))/accumulation
                scaler.scale(loss).backward()
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        model.eval()
        validation = torch.zeros(2, device=device)
        with torch.no_grad():
            for x, y in valid_loader:
                with autocast(device):
                    value = reconstruction_loss(model(x.to(device)), y.to(device), config.get('loss_norm', 'L1'))
                validation += torch.tensor([value.item()*len(x), len(x)], device=device)
        if distributed:
            dist.all_reduce(validation)
        value = (validation[0]/validation[1]).item()
        improved = value < best
        best = min(best, value)
        patience = 0 if improved else patience+1
        if rank == 0:
            state = (model.module if distributed else model).state_dict()
            if improved:
                torch.save(dict(model=state, config=config, epoch=epoch+1, val_loss=value), out/'best_model.pth')
            if (epoch+1)%5 == 0:
                torch.save(dict(model=state, config=config, epoch=epoch+1, best=best, optim=optimizer.state_dict(), sched=scheduler.state_dict()), out/'resume_checkpoint.pth')
            print(f'Epoch {epoch+1}: validation={value:.8g}, best={best:.8g}', flush=True)
        if patience >= config['early_stopping_patience']:
            break
    if distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.resume)


if __name__ == '__main__':
    main()
