# -*- coding: utf-8 -*-
"""
HW2 Phoneme Classification - RNN/LSTM Version

This script is an updated version of the baseline, replacing the MLP
with a more powerful Bidirectional LSTM model for sequence modeling.
"""

import os
import gc
import math
import random
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


# -----------------------------
# Reproducibility
# -----------------------------
def same_seeds(seed: int = 1213):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------
# Feature helpers (Original functions are kept for reference)
# -----------------------------
def load_feat(path: str) -> torch.Tensor:
    return torch.load(path)


def read_usage_list(split_file: str) -> list:
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]


# MODIFIED: Data preprocessing now prepares data as sequences for the RNN
def preprocess_data_rnn(
        split: str,
        feat_dir: str,
        phone_path: str,
        train_ratio: float = 0.75,
        random_seed: int = 1213,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """
    Prepares data for RNNs by returning lists of sequences.

    Returns:
      X_list: A list of tensors, where each tensor is an utterance [T, 39]
      y_list: A list of tensors, where each tensor is the labels for an utterance [T]
    """
    class_num = 41
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val', or 'test'")

    if split in ("train", "val"):
        mode = "train"
        usage_list_all = read_usage_list(os.path.join(phone_path, "train_split.txt"))
        random.seed(random_seed)
        random.shuffle(usage_list_all)
        train_len = int(len(usage_list_all) * train_ratio)
        usage_list = usage_list_all[:train_len] if split == "train" else usage_list_all[train_len:]
        label_dict = {}
        with open(os.path.join(phone_path, f"{mode}_labels.txt"), "r") as f:
            for line in f:
                p = line.strip().split()
                label_dict[p[0]] = [int(v) for v in p[1:]]
    else:
        mode = "test"
        usage_list = read_usage_list(os.path.join(phone_path, "test_split.txt"))
        label_dict = None

    print(f"[Dataset] #classes={class_num}, #utterances for {split}: {len(usage_list)}")

    X_list = []
    y_list = [] if mode in ("train", "val") else None

    feat_mode_dir = os.path.join(feat_dir, mode)
    for fname in tqdm(usage_list):
        feat = load_feat(os.path.join(feat_mode_dir, f"{fname}.pt"))
        X_list.append(feat)
        if mode in ("train", "val"):
            labels = torch.LongTensor(label_dict[fname])
            y_list.append(labels)

    return X_list, y_list


# -----------------------------
# Dataset & Collate Function (MODIFIED for RNN)
# -----------------------------
class LibriSeqDataset(Dataset):
    def __init__(self, X: List[torch.Tensor], y: Optional[List[torch.Tensor]] = None):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            # Need to return a dummy label for test set to make collate_fn work
            return self.data[idx], None

    def __len__(self):
        return len(self.data)


def collate_fn_pad(batch):
    features, labels = zip(*batch)

    # Pad features
    padded_features = pad_sequence(features, batch_first=True)

    # Pad labels if they exist (i.e., not in test mode)
    if labels[0] is not None:
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    else:
        padded_labels = None

    return padded_features, padded_labels


# -----------------------------
# Model (NEW: RNNClassifier)
# -----------------------------
class RNNClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 41,
            hidden_layers: int = 4,
            hidden_dim: int = 256,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if hidden_layers > 1 else 0,
        )
        self.head = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        outputs, _ = self.rnn(x)  # outputs shape: (batch, seq_len, hidden_dim * 2)
        logits = self.head(outputs)  # logits shape: (batch, seq_len, output_dim)
        return logits


# -----------------------------
# Train / Eval (MODIFIED for RNN)
# -----------------------------
def run_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        use_amp: bool = False,
        train: bool = True,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)
    with torch.set_grad_enabled(train):
        for features, labels in pbar:
            features = features.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(features)  # output shape: (B, T, C)
                if labels is not None:
                    # Reshape for CrossEntropyLoss: (B*T, C) and (B*T)
                    loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
                else:
                    loss = None

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if labels is not None:
                with torch.no_grad():
                    preds = outputs.argmax(dim=-1)
                    mask = (labels != -1)  # Ignore padded labels (-1)
                    total_correct += (preds[mask] == labels[mask]).sum().item()
                    total_samples += mask.sum().item()
                    total_loss += loss.item()

            if labels is not None:
                pbar.set_description(f"{'Train' if train else 'Val'} "
                                     f"acc={(total_correct / max(1, total_samples)):.4f} "
                                     f"loss={(total_loss / max(1, (pbar.n + 1))):.4f}")

    avg_loss = total_loss / max(1, len(loader))
    avg_acc = total_correct / max(1, total_samples)
    return avg_acc, avg_loss


def train_and_eval(args):
    same_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # MODIFIED: Use RNN data prep
    train_X, train_y = preprocess_data_rnn(
        split="train", feat_dir=args.feat_dir, phone_path=args.phone_path,
        train_ratio=args.train_ratio, random_seed=args.seed
    )
    val_X, val_y = preprocess_data_rnn(
        split="val", feat_dir=args.feat_dir, phone_path=args.phone_path,
        train_ratio=args.train_ratio, random_seed=args.seed
    )

    # MODIFIED: Use sequence dataset and collate function
    train_ds = LibriSeqDataset(train_X, train_y)
    val_ds = LibriSeqDataset(val_X, val_y)
    del train_X, train_y, val_X, val_y
    gc.collect()

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn_pad
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_fn_pad
    )

    # MODIFIED: Instantiate RNNClassifier
    # Input dim is 39 (original feature dim), not concatenated
    model = RNNClassifier(
        input_dim=39, output_dim=41, hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)

    # MODIFIED: Loss must ignore padding index
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_acc, train_loss = run_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=args.amp, train=True, scaler=scaler
        )
        with torch.no_grad():
            val_acc, val_loss = run_one_epoch(
                model, val_loader, criterion, optimizer=None, device=device, use_amp=False, train=False
            )

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        print(f"[{epoch:03d}] Train Acc: {train_acc:.5f} Loss: {train_loss:.5f} | "
              f"Val Acc: {val_acc:.5f} Loss: {val_loss:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"Saved best model to {args.model_path} (Val Acc={best_val_acc:.5f})")

    del train_ds, val_ds, train_loader, val_loader
    gc.collect()


@torch.no_grad()
def predict_and_write_csv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Loading best model from {args.model_path}")

    # MODIFIED: Load test data as sequences
    test_X, _ = preprocess_data_rnn(
        split="test", feat_dir=args.feat_dir, phone_path=args.phone_path
    )
    test_ds = LibriSeqDataset(test_X, None)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_pad
    )

    # MODIFIED: Instantiate RNN model for prediction
    model = RNNClassifier(
        input_dim=39, output_dim=41, hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    # MODIFIED: Handle sequential prediction
    for features, _ in tqdm(test_loader, desc="Predict", leave=False):
        features = features.to(device)
        outputs = model(features)  # (B, T, C)

        # We need to get predictions for each original, unpadded frame
        # We can find original lengths by seeing where the features are non-zero
        # A simpler way (though less robust) is to assume zero-padding
        for i in range(features.shape[0]):
            # Find the actual length of the sequence before padding
            actual_len = (features[i].abs().sum(dim=1) != 0).sum().item()
            pred = outputs[i, :actual_len, :].argmax(dim=-1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    out_csv = args.out_csv
    with open(out_csv, "w") as f:
        f.write("Id,Class\n")
        for i, y in enumerate(preds):
            f.write(f"{i},{y}\n")
    print(f"[Predict] Wrote predictions to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="HW2 Phoneme Classification - RNN/LSTM Version")
    # paths
    p.add_argument("--feat_dir", type=str, default="./libriphone/feat", help="directory of precomputed features")
    p.add_argument("--phone_path", type=str, default="./libriphone", help="path containing split/labels files")
    p.add_argument("--model_path", type=str, default="./model_rnn.ckpt", help="where to save/load model")
    p.add_argument("--out_csv", type=str, default="./prediction_rnn.csv", help="prediction output CSV path")

    # data / split
    p.add_argument("--train_ratio", type=float, default=0.75, help="train/val split ratio on train_split.txt")

    # model
    p.add_argument("--hidden_layers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)

    # train
    p.add_argument("--seed", type=int, default=1213)
    p.add_argument("--batch_size", type=int, default=32)  # RNNs are more memory intensive, may need smaller batch size
    p.add_argument("--epochs", type=int, default=20)  # May need more epochs
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])

    # flags
    p.add_argument("--amp", action="store_true", help="enable mixed precision training")

    args = p.parse_args()
    if args.scheduler == "none":
        args.scheduler = None
    return args


def main():
    args = parse_args()
    train_and_eval(args)
    predict_and_write_csv(args)


if __name__ == "__main__":
    main()