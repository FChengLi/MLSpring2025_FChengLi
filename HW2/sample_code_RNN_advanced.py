# -*- coding: utf-8 -*-
"""
HW2 Phoneme Classification - Advanced RNN/LSTM Version

此脚本是RNN版本的升级版，融合了帧拼接技术，
并使用了一个更深、更宽、包含LayerNorm和更强分类头的先进模型架构。
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
# 可复现性
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
# 特征处理辅助函数 (为帧拼接而添加)
# -----------------------------
def load_feat(path: str) -> torch.Tensor:
    return torch.load(path)


def read_usage_list(split_file: str) -> list:
    with open(split_file, "r") as f:
        return [line.strip() for line in f.readlines()]


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
    """拼接过去/未来的帧，同时保持序列长度不变。"""
    assert concat_n % 2 == 1, "concat_nframes 必须是奇数"
    if concat_n < 2:
        return x
    seq_len, feat_dim = x.size(0), x.size(1)
    x_rep = x.repeat(1, concat_n).view(seq_len, concat_n, feat_dim).permute(1, 0, 2)
    mid = concat_n // 2
    for r in range(1, mid + 1):
        x_rep[mid + r, :] = shift(x_rep[mid + r], r)
        x_rep[mid - r, :] = shift(x_rep[mid - r], -r)
    return x_rep.permute(1, 0, 2).reshape(seq_len, concat_n * feat_dim)


# -----------------------------
# 数据预处理 (MODIFIED: 加入了帧拼接功能)
# -----------------------------
def preprocess_data_seq(
        split: str,
        feat_dir: str,
        phone_path: str,
        concat_nframes: int,
        train_ratio: float = 0.75,
        random_seed: int = 1213,
) -> Tuple[List[torch.Tensor], Optional[List[torch.Tensor]]]:
    """为序列模型准备数据，返回序列列表。"""
    class_num = 41
    if split not in ("train", "val", "test"):
        raise ValueError("split 必须是 'train', 'val', 或 'test'")

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

    print(f"[数据] 类别数={class_num}, 用于'{split}'的语音片段数: {len(usage_list)}")

    X_list = []
    y_list = [] if mode in ("train", "val") else None

    feat_mode_dir = os.path.join(feat_dir, mode)
    for fname in tqdm(usage_list):
        feat = load_feat(os.path.join(feat_mode_dir, f"{fname}.pt"))

        # 【核心修改】在这里应用帧拼接
        feat = concat_feat(feat, concat_nframes)

        X_list.append(feat)
        if mode in ("train", "val"):
            labels = torch.LongTensor(label_dict[fname])
            y_list.append(labels)

    return X_list, y_list


# -----------------------------
# 数据集与 Collate 函数 (与之前RNN版本相同)
# -----------------------------
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
# 模型 (NEW: AdvancedRNNClassifier)
# -----------------------------
class AdvancedRNNClassifier(nn.Module):
    """
    一个更先进的RNN分类器，融合了更深的网络结构、LayerNorm和更强大的分类头。
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int = 41,
            hidden_layers: int = 6,
            hidden_dim: int = 512,
            dropout: float = 0.4,
    ):
        super().__init__()

        # --- LSTM 主干网络 ---
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=hidden_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if hidden_layers > 1 else 0,
        )

        # --- 分类头 ---
        self.head = nn.Sequential(
            # LayerNorm 对RNN更友好，用于稳定训练
            nn.LayerNorm(hidden_dim * 2),
            # 一个中间处理层，增加模型的非线性能力
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 最终输出层
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x 的形状: (batch, seq_len, input_dim)
        # 1. 通过 LSTM 主干网络
        rnn_outputs, _ = self.rnn(x)
        # 2. 通过强大的分类头
        logits = self.head(rnn_outputs)
        return logits


# -----------------------------
# 训练与评估 (与之前RNN版本相同)
# -----------------------------
def run_one_epoch(
        model: nn.Module, loader: DataLoader, criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer], device: torch.device,
        use_amp: bool = False, train: bool = True,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
):
    model.train() if train else model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    pbar = tqdm(loader, leave=False)

    with torch.set_grad_enabled(train):
        for features, labels in pbar:
            features = features.to(device, non_blocking=True)
            if labels is not None:
                labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(features)
                if labels is not None:
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
                    mask = (labels != -1)
                    total_correct += (preds[mask] == labels[mask]).sum().item()
                    total_samples += mask.sum().item()
                    total_loss += loss.item()

                pbar.set_description(
                    f"{'训练' if train else '验证'} "
                    f"准确率={(total_correct / max(1, total_samples)):.4f} "
                    f"损失={(total_loss / max(1, (pbar.n + 1))):.4f}"
                )

    avg_acc = total_correct / max(1, total_samples)
    avg_loss = total_loss / max(1, len(loader))
    return avg_acc, avg_loss


def train_and_eval(args):
    same_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 使用包含帧拼接功能的数据预处理
    train_X, train_y = preprocess_data_seq(
        "train", args.feat_dir, args.phone_path, args.concat_nframes, args.train_ratio, args.seed
    )
    val_X, val_y = preprocess_data_seq(
        "val", args.feat_dir, args.phone_path, args.concat_nframes, args.train_ratio, args.seed
    )

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

    # 【核心修改】计算新的输入维度并实例化 AdvancedRNNClassifier 模型
    input_dim = 39 * args.concat_nframes
    model = AdvancedRNNClassifier(
        input_dim=input_dim, output_dim=41, hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_acc, train_loss = run_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=args.amp, train=True, scaler=scaler
        )
        with torch.no_grad():
            val_acc, val_loss = run_one_epoch(
                model, val_loader, criterion, None, device, use_amp=False, train=False
            )

        scheduler.step(val_loss)

        print(f"[{epoch:03d}] 训练 准确率: {train_acc:.5f} 损失: {train_loss:.5f} | "
              f"验证 准确率: {val_acc:.5f} 损失: {val_loss:.5f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_path)
            print(f"已保存最佳模型到 {args.model_path} (验证准确率={best_val_acc:.5f})")

    del train_ds, val_ds, train_loader, val_loader, model
    gc.collect()


@torch.no_grad()
def predict_and_write_csv(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[预测] 从 {args.model_path} 加载最佳模型")

    test_X, _ = preprocess_data_seq("test", args.feat_dir, args.phone_path, args.concat_nframes)
    test_ds = LibriSeqDataset(test_X, None)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_pad
    )

    input_dim = 39 * args.concat_nframes
    model = AdvancedRNNClassifier(
        input_dim=input_dim, output_dim=41, hidden_layers=args.hidden_layers,
        hidden_dim=args.hidden_dim, dropout=args.dropout
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    for features, _ in tqdm(test_loader, desc="预测中", leave=False):
        features = features.to(device)
        outputs = model(features)

        for i in range(features.shape[0]):
            actual_len = (features[i].abs().sum(dim=1) != 0).sum().item()
            pred = outputs[i, :actual_len, :].argmax(dim=-1).cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    with open(args.out_csv, "w") as f:
        f.write("Id,Class\n")
        for i, y in enumerate(preds):
            f.write(f"{i},{y}\n")
    print(f"[预测] 已将预测结果写入 {args.out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="HW2 音素分类 - 高级RNN/LSTM版本")
    # 路径参数
    p.add_argument("--feat_dir", default="./libriphone/feat", help="预计算特征的目录")
    p.add_argument("--phone_path", default="./libriphone", help="包含切分/标签文件的路径")
    p.add_argument("--model_path", default="./model_advanced_rnn.ckpt", help="模型保存/加载路径")
    p.add_argument("--out_csv", default="./prediction_advanced_rnn.csv", help="预测输出CSV文件的路径")

    # 数据参数
    p.add_argument("--concat_nframes", type=int, default=3, help="为RNN输入特征而拼接的帧数 (必须是奇数)")
    p.add_argument("--train_ratio", type=float, default=0.75, help="训练集/验证集的切分比例")

    # 模型参数
    p.add_argument("--hidden_layers", type=int, default=6, help="RNN/LSTM 的层数")
    p.add_argument("--hidden_dim", type=int, default=512, help="RNN/LSTM 隐藏层的维度")
    p.add_argument("--dropout", type=float, default=0.4, help="Dropout 的比率")

    # 训练参数
    p.add_argument("--seed", type=int, default=1213)
    p.add_argument("--batch_size", type=int, default=32, help="批次大小")
    p.add_argument("--epochs", type=int, default=25, help="训练周期数")
    p.add_argument("--lr", type=float, default=5e-4, help="学习率")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    p.add_argument("--num_workers", type=int, default=4, help="数据加载器使用的工作进程数")

    # 标志参数
    p.add_argument("--amp", action="store_true", help="启用混合精度训练 (AMP)")

    args = p.parse_args()
    if args.concat_nframes % 2 == 0:
        raise ValueError("--concat_nframes 必须是奇数 (例如 3, 5, 7)")
    return args


def main():
    args = parse_args()
    train_and_eval(args)
    predict_and_write_csv(args)


if __name__ == "__main__":
    main()