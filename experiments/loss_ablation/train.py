import argparse
import json
from pathlib import Path

from vivivit.loss_ablation import build_model
from vivivit.train import train

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
train(json.loads(Path(args.config).read_text()), model_factory=build_model)
