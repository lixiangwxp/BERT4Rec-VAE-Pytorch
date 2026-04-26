import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataloaders import dataloader_factory
from models import model_factory
from trainers import trainer_factory


def main():
    parser = argparse.ArgumentParser(description='Test a trained BERT4Rec checkpoint and run one forward check.')
    parser.add_argument('--experiment', required=True, help='Experiment folder containing config.json and models/')
    parser.add_argument('--device', default=None, choices=['mps', 'cuda', 'cpu'])
    args_cli = parser.parse_args()

    export_root = Path(args_cli.experiment)
    with export_root.joinpath('config.json').open() as f:
        config = json.load(f)

    args = SimpleNamespace(**config)
    if args_cli.device is not None:
        args.device = args_cli.device
    args.num_gpu = 0 if args.device != 'cuda' else args.num_gpu

    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, str(export_root))
    trainer.test()

    seqs, candidates, _ = next(iter(test_loader))
    seqs = seqs.to(trainer.device)
    candidates = candidates.to(trainer.device)
    with torch.no_grad():
        scores = trainer.model(seqs)[:, -1, :].gather(1, candidates)
        top_positions = scores[0].topk(min(5, scores.size(1))).indices
        top_items = candidates[0].gather(0, top_positions).cpu().tolist()

    print('REAL_DATA_INFERENCE_RESULT=' + json.dumps({
        'device': str(trainer.device),
        'checkpoint': str(export_root.joinpath('models', 'best_acc_model.pth')),
        'score_shape': list(scores.shape),
        'top_items_for_first_user': top_items,
    }, indent=2))


if __name__ == '__main__':
    main()
