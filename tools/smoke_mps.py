import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import model_factory
from trainers import trainer_factory
from utils import setup_train


def build_args(cli_args):
    return SimpleNamespace(
        mode='train',
        template=None,
        trainer_code='bert',
        device=cli_args.device,
        num_gpu=1,
        device_idx='0',
        optimizer='Adam',
        lr=0.001,
        weight_decay=0.0,
        momentum=None,
        enable_lr_schedule=False,
        decay_step=15,
        gamma=0.1,
        num_epochs=cli_args.num_epochs,
        train_batch_size=4,
        val_batch_size=4,
        test_batch_size=4,
        log_period_as_iter=10**9,
        metric_ks=[1, 5, 10],
        best_metric='NDCG@10',
        model_code='bert',
        model_init_seed=0,
        num_items=cli_args.num_items,
        bert_num_items=cli_args.num_items,
        bert_dropout=0.1,
        bert_hidden_units=cli_args.hidden_units,
        bert_mask_prob=0.2,
        bert_max_len=cli_args.max_len,
        bert_num_blocks=1,
        bert_num_heads=2,
        experiment_dir=cli_args.experiment_dir,
        experiment_description=cli_args.experiment_description,
    )


def build_loaders(args):
    mask_token = args.num_items + 1
    train_sequences = []
    train_labels = []
    eval_sequences = []
    candidates = []
    candidate_labels = []

    for user in range(16):
        sequence = (torch.arange(args.bert_max_len) + user) % args.num_items + 1
        labels = torch.zeros(args.bert_max_len, dtype=torch.long)
        for offset in (1, 3):
            labels[-offset] = sequence[-offset]
            sequence[-offset] = mask_token
        train_sequences.append(sequence)
        train_labels.append(labels)

        answer = int((user * 3) % args.num_items + 1)
        negatives = []
        item = answer + 1
        while len(negatives) < 5:
            normalized = (item - 1) % args.num_items + 1
            if normalized != answer:
                negatives.append(normalized)
            item += 1
        eval_sequence = (torch.arange(args.bert_max_len) + user + 5) % args.num_items + 1
        eval_sequence[-1] = mask_token
        eval_sequences.append(eval_sequence)
        candidates.append(torch.tensor([answer] + negatives, dtype=torch.long))
        candidate_labels.append(torch.tensor([1] + [0] * len(negatives), dtype=torch.long))

    train_dataset = TensorDataset(torch.stack(train_sequences), torch.stack(train_labels))
    eval_dataset = TensorDataset(torch.stack(eval_sequences), torch.stack(candidates), torch.stack(candidate_labels))

    pin_memory = args.device == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader


def run_inference_check(trainer, test_loader, export_root):
    checkpoint_path = Path(export_root).joinpath('models', 'best_acc_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
    target_model = trainer.model.module if trainer.is_parallel else trainer.model
    target_model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()

    seqs, batch_candidates, _ = next(iter(test_loader))
    seqs = seqs.to(trainer.device)
    batch_candidates = batch_candidates.to(trainer.device)
    with torch.no_grad():
        scores = trainer.model(seqs)[:, -1, :].gather(1, batch_candidates)
        top_positions = scores[0].topk(min(5, scores.size(1))).indices
        top_items = batch_candidates[0].gather(0, top_positions).cpu().tolist()

    return {
        'device': str(trainer.device),
        'checkpoint': str(checkpoint_path),
        'score_shape': list(scores.shape),
        'top_items_for_first_user': top_items,
    }


def main():
    parser = argparse.ArgumentParser(description='Run a tiny BERT4Rec train + inference check.')
    parser.add_argument('--device', default='auto', choices=['auto', 'mps', 'cuda', 'cpu'])
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--num-items', type=int, default=20)
    parser.add_argument('--max-len', type=int, default=8)
    parser.add_argument('--hidden-units', type=int, default=32)
    parser.add_argument('--experiment-dir', default='experiments')
    parser.add_argument('--experiment-description', default='smoke-mps')
    cli_args = parser.parse_args()

    args = build_args(cli_args)
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = build_loaders(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()
    trainer.test()
    result = run_inference_check(trainer, test_loader, export_root)
    print('SMOKE_INFERENCE_RESULT=' + json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
