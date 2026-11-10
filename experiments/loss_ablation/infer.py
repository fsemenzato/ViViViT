import argparse
import torch

from vivivit.infer import infer
from vivivit.loss_ablation import build_model

parser = argparse.ArgumentParser()
for name in ['checkpoint', 'data', 'output']:
    parser.add_argument('--'+name, required=True)
args = parser.parse_args()
infer(args.checkpoint, args.data, args.output, range(1086, 1113), 'cuda' if torch.cuda.is_available() else 'cpu', build_model)
