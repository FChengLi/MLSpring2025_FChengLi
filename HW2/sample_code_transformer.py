# -*- coding: utf-8 -*-
"""
HW2 Phoneme Classification - Transformer Version

This script uses a Transformer Encoder model, the state-of-the-art
architecture for sequence modeling tasks.
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
# Data Loading (Identical to RNN version)
# -----------------------------
def load_feat(path: str) -> torch.Tensor:
    return torch.load(path)


def read_usage_list(split_file: str) -> list:
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_data_seq(
        split: str,
        feat_dir: str,
        phone_path: str,
        train_ratio: float = 0.75,
        random_seed: int = 1213,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
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
    X_list, y_list = [], [] if mode in ("train", "val") else None
    feat_mode_dir = os.path.join(feat_dir, mode)
    for fname in tqdm(usage_list):
        feat = load_feat(os.path.join(feat_mode_dir, f"{fname}.pt"))
        X_list.append(feat)
        if mode in ("train", "val"):
            labels = torch.LongTensor(label_dict[fname])
            y_list.append(labels)

    return X_list, y_list


class LibriSeqDataset(Dataset):
    def __init__(self, X: List[torch.Tensor], y: Optional[List[torch.Tensor]] = None):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx], None

    def __len__(self):
        return len(self.data)


def collate_fn_pad(batch):
    features, labels = zip(*batch)
    padded_features = pad_sequence(features, batch_first=True)
    if labels[0] is not None:
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    else:
        padded_labels = None
    return padded_features, padded_labels


# -----------------------------
# Model (NEW: TransformerClassifier)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Transpose for positional encoding: [seq_len, batch_size, embedding_dim]
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        x = x.transpose(0, 1)
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = 41,
            d_model: int = 256,
            nhead: int = 8,
            num_encoder_layers: int = 4,
            dim_feedforward: int = 1024,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Makes handling batches easier
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: (B, T, D_in)
            src_key_padding_mask: (B, T) # Tells the model which tokens are padding
        """
        # 1. Project input features to d_model
        src = self.input_proj(src)  # (B, T, d_model)

        # 2. Add positional encoding
        src = self.pos_encoder(src)

        # 3. Pass through Transformer Encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # 4. Final classification head
        logits = self.head(output)  # (B, T, C)
        return logits


# -----------------------------
# Train / Eval (Mostly unchanged from RNN version)
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

    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, leave=False)

    with torch.set_grad_enabled(train):
        for features, labels in pbar:
            features = features.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)

            # Create padding mask for Transformer
            # The mask should be True for padded values and False for real values
            padding_mask = (features.abs().sum(dim=-1) == 0) if features is not None else None

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(features, src_key_padding_mask=padding_mask)
                if labels is not None:
                    loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if labels is not None:
                with torch.no_grad():
                    preds = outputs.argmax(dim=-1)
                    label_mask = (labels != -1)
                    total_correct += (preds[label_mask] == labels[label_mask]).sum().item()
                    total_samples += label_mask.sum().item()
                    total_loss += loss.item()

                pbar.set_description(
                    f"{'Train' if train else 'Val'} "
                    f"acc={(total_correct / max(1, total_samples)):.4f} "
                    f"loss={(total_loss / max(1, (pbar.n + 1))):.4f}"
                )

    return total_correct / max(1, total_samples), total_loss / max(1, len(loader))


def train_and_eval(args):
    same_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_X, train_y = preprocess_data_seq("train", args.feat_dir, args.phone_path, args.train_ratio, args.seed)
    val_X, val_y = preprocess_data_seq("val", args.feat_dir, args.phone_path, args.train_ratio, args.seed)

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

    # NEW: Instantiate TransformerClassifier
    model = TransformerClassifier(
        input_dim=39, output_dim=41, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers, dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_acc, train_loss = run_one_epoch(model, train_loader, criterion, optimizer, device, args.amp, True, scaler)
        with torch.no_grad():
            val_acc, val_loss = run_one_epoch(model, val_loader, criterion, None, device, args.amp, False)

        scheduler.step(val_loss)
        print(
            f"[{epoch:03d}] Train Acc: {train_acc:.5f} Loss: {train_loss:.5f} | Val Acc: {val_acc:.5f} Loss: {val_loss:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"Saved best model to {args.model_path} (Val Acc={best_val_acc:.5f})")

    del train_ds, val_ds, train_loader, val_loader, model
    gc.collect()


@torch.no_grad()
def predict_and_write_csv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Loading best model from {args.model_path}")

    test_X, _ = preprocess_data_seq("test", args.feat_dir, args.phone_path)
    test_ds = LibriSeqDataset(test_X, None)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=(device.type == "cuda"), collate_fn=collate_fn_pad)

    model = TransformerClassifier(
        input_dim=39, output_dim=41, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers, dim_feedforward=args.dim_feedforward
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    for features, _ in tqdm(test_loader, desc="Predict", leave=False):
        features = features.to(device)
        padding_mask = (features.abs().sum(dim=-1) == 0)
        outputs = model(features, src_key_padding_mask=padding_mask)

        for i in range(features.shape[0]):
            actual_len = (~padding_mask[i]).sum().item()
            pred = outputs[i, :actual_len, :].argmax(dim=-1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    with open(args.out_csv, "w") as f:
        f.write("Id,Class\n")
        for i, y in enumerate(preds):
            f.write(f"{i},{y}\n")
    print(f"[Predict] Wrote predictions to {args.out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="HW2 Phoneme Classification - Transformer Version")
    # Paths
    p.add_argument("--feat_dir", default="./libriphone/feat")
    p.add_argument("--phone_path", default="./libriphone")
    p.add_argument("--model_path", default="./model_transformer.ckpt")
    p.add_argument("--out_csv", default="./prediction_transformer.csv")

    # Data
    p.add_argument("--train_ratio", type=float, default=0.75)

    # Model - Transformer Specific
    p.add_argument("--d_model", type=int, default=256, help="Transformer model dimension")
    p.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    p.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers")
    p.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of feedforward network")

    # Training
    p.add_argument("--seed", type=int, default=1213)
    p.add_argument("--batch_size", type=int, default=16)  # Transformers are memory hungry
    p.add_argument("--epochs", type=int, default=30)  # May need more epochs to converge
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=4)

    # Flags
    p.add_argument("--amp", action="store_true", help="enable mixed precision training")

    args = p.parse_args()
    return args


def main():
    args = parse_args()
    train_and_eval(args)
    predict_and_write_csv(args)


if __name__ == "__main__":
    main()