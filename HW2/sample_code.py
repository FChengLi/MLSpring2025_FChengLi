# -*- coding: utf-8 -*-
"""
HW2 Phoneme Classification - Stronger Baseline (Single File)

Usage:
  python hw2_phoneme_classifier.py \
    --feat_dir ./libriphone/feat \
    --phone_path ./libriphone \
    --concat_nframes 21 \
    --hidden_layers 4 \
    --hidden_dim 256 \
    --dropout 0.3 \
    --batch_size 512 \
    --epochs 15 \
    --lr 1e-3 \
    --model_path ./model.ckpt \
    --amp

Data layout (same as original):
- libriphone/train_split.txt
- libriphone/train_labels.txt
- libriphone/test_split.txt
- libriphone/feat/train/*.pt
- libriphone/feat/test/*.pt
"""

import os
import gc
import math
import random
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
# Feature helpers
# -----------------------------
def load_feat(path: str) -> torch.Tensor:
    return torch.load(path)


def shift(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return x
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
        return torch.cat((left, right), dim=0)
    else:
        right = x[-1].repeat(n, 1)
        left = x[n:]
        return torch.cat((left, right), dim=0)


def concat_feat(x: torch.Tensor, concat_n: int) -> torch.Tensor:
    """Concat past/future frames, keeping length unchanged."""
    assert concat_n % 2 == 1, "concat_nframes must be odd"
    if concat_n < 2:
        return x
    seq_len, feat_dim = x.size(0), x.size(1)
    x_rep = x.repeat(1, concat_n).view(seq_len, concat_n, feat_dim).permute(1, 0, 2)
    mid = concat_n // 2
    for r in range(1, mid + 1):
        x_rep[mid + r, :] = shift(x_rep[mid + r], r)
        x_rep[mid - r, :] = shift(x_rep[mid - r], -r)
    return x_rep.permute(1, 0, 2).reshape(seq_len, concat_n * feat_dim)


def read_usage_list(split_file: str) -> list:
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_data(
    split: str,
    feat_dir: str,
    phone_path: str,
    concat_nframes: int,
    train_ratio: float = 0.75,
    random_seed: int = 1213,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      X: [N_frames, 39 * concat_nframes]
      y: [N_frames] (for train/val) or None (for test)
    """
    class_num = 41
    if split not in ("train", "val", "test"):
        raise ValueError("split must be 'train', 'val', or 'test'")

    # Figure out mode & usage list
    if split in ("train", "val"):
        mode = "train"
        usage_list_all = read_usage_list(os.path.join(phone_path, "train_split.txt"))
        random.seed(random_seed)
        random.shuffle(usage_list_all)
        train_len = int(len(usage_list_all) * train_ratio)
        usage_list = usage_list_all[:train_len] if split == "train" else usage_list_all[train_len:]
        # label dict
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
    y_list = [] if mode == "train" else None

    feat_mode_dir = os.path.join(feat_dir, mode)
    for fname in tqdm(usage_list):
        feat = load_feat(os.path.join(feat_mode_dir, f"{fname}.pt"))  # [T, 39]
        T = feat.shape[0]
        feat_cat = concat_feat(feat, concat_nframes)  # [T, 39*concat_nframes]
        X_list.append(feat_cat)
        if mode == "train":
            labels = torch.LongTensor(label_dict[fname])  # length T
            assert labels.numel() == T, "Label length must match frames"
            y_list.append(labels)

    X = torch.cat(X_list, dim=0)
    if mode == "train":
        y = torch.cat(y_list, dim=0)
        print(f"[INFO] {split} set X={tuple(X.shape)}, y={tuple(y.shape)}")
        return X, y
    else:
        print(f"[INFO] {split} set X={tuple(X.shape)}")
        return X, None


# -----------------------------
# Dataset
# -----------------------------
class LibriDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: Optional[torch.Tensor] = None):
        self.data = X
        self.label = y if y is not None else None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


# -----------------------------
# Model
# -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class Classifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 41,
        hidden_layers: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = [BasicBlock(input_dim, hidden_dim, dropout)]
        for _ in range(hidden_layers - 1):
            layers.append(BasicBlock(hidden_dim, hidden_dim, dropout))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

        # Kaiming init for Linear
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# -----------------------------
# Train / Eval
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
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                features, labels = batch
                labels = labels.to(device, non_blocking=True)
            else:
                features = batch
                labels = None

            features = features.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features)
                    loss = criterion(outputs, labels) if labels is not None else None
            else:
                outputs = model(features)
                loss = criterion(outputs, labels) if labels is not None else None

            if train:
                scaler = scaler if scaler is not None else torch.cuda.amp.GradScaler(enabled=use_amp)
                if use_amp:
                    scaler.scale(loss).step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # metrics
            if labels is not None:
                with torch.no_grad():
                    preds = outputs.argmax(dim=1)
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    total_loss += loss.item()

            if labels is not None:
                pbar.set_description(f"{'Train' if train else 'Val'} "
                                     f"acc={(total_correct/max(1,total_samples)):.4f} "
                                     f"loss={(total_loss/max(1,(pbar.n+1))):.4f}")

    avg_loss = total_loss / max(1, len(loader))
    avg_acc = total_correct / max(1, total_samples)
    return avg_acc, avg_loss


def train_and_eval(
    args,
):
    same_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # preprocess
    train_X, train_y = preprocess_data(
        split="train",
        feat_dir=args.feat_dir,
        phone_path=args.phone_path,
        concat_nframes=args.concat_nframes,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )
    val_X, val_y = preprocess_data(
        split="val",
        feat_dir=args.feat_dir,
        phone_path=args.phone_path,
        concat_nframes=args.concat_nframes,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )

    # datasets / loaders
    train_ds = LibriDataset(train_X, train_y)
    val_ds = LibriDataset(val_X, val_y)

    # Free raw tensors to save RAM (optional)
    del train_X, train_y, val_X, val_y
    gc.collect()

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    input_dim = 39 * args.concat_nframes
    model = Classifier(
        input_dim=input_dim,
        output_dim=41,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_acc, train_loss = run_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=args.amp, train=True, scaler=scaler
        )
        with torch.no_grad():
            val_acc, val_loss = run_one_epoch(
                model, val_loader, criterion, optimizer=None, device=device, use_amp=False, train=False
            )

        if scheduler is not None:
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

    # free
    del train_ds, val_ds, train_loader, val_loader
    gc.collect()


@torch.no_grad()
def predict_and_write_csv(
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] Loading best model from {args.model_path}")

    # load test
    test_X, _ = preprocess_data(
        split="test",
        feat_dir=args.feat_dir,
        phone_path=args.phone_path,
        concat_nframes=args.concat_nframes,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
    )
    test_ds = LibriDataset(test_X, None)
    pin = device.type == "cuda"
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    input_dim = 39 * args.concat_nframes
    model = Classifier(
        input_dim=input_dim,
        output_dim=41,
        hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    for batch in tqdm(test_loader, desc="Predict", leave=False):
        features = batch.to(device, non_blocking=True)
        outputs = model(features)
        pred = outputs.argmax(dim=1).cpu().numpy()
        preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    out_csv = args.out_csv
    with open(out_csv, "w") as f:
        f.write("Id,Class\n")
        for i, y in enumerate(preds):
            f.write(f"{i},{y}\n")
    print(f"[Predict] Wrote predictions to {out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="HW2 Phoneme Classification - Stronger Baseline (Single File)")
    # paths
    p.add_argument("--feat_dir", type=str, default="./libriphone/feat", help="directory of precomputed features")
    p.add_argument("--phone_path", type=str, default="./libriphone", help="path containing split/labels files")
    p.add_argument("--model_path", type=str, default="./model.ckpt", help="where to save/load model")
    p.add_argument("--out_csv", type=str, default="./prediction.csv", help="prediction output CSV path")

    # data / split
    p.add_argument("--concat_nframes", type=int, default=21, help="odd number of frames to concat (2k+1)")
    p.add_argument("--train_ratio", type=float, default=0.75, help="train/val split ratio on train_split.txt")

    # model
    p.add_argument("--hidden_layers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)

    # train
    p.add_argument("--seed", type=int, default=1213)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau", "cosine"])

    # flags
    p.add_argument("--amp", action="store_true", help="enable mixed precision training")

    args = p.parse_args()

    if args.concat_nframes % 2 == 0:
        raise ValueError("--concat_nframes must be odd (e.g., 3, 11, 21)")

    if args.scheduler == "none":
        args.scheduler = None

    return args


def main():
    args = parse_args()
    train_and_eval(args)
    predict_and_write_csv(args)


if __name__ == "__main__":
    main()
