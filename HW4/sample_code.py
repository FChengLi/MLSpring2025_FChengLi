# -*- coding: utf-8 -*-
"""
完整可运行代码：结合了 Conformer 和 Self-Attention Pooling 的说话人识别模型 (已修复推理显存问题)
"""

# ===================================================================
# Part 1: 环境设置与通用函数
# ===================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import json
import math
import csv
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seed(2023)


# ===================================================================
# Part 2: 数据集与数据加载器
# ===================================================================
class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        mapping_path = Path(data_dir) / 'mapping.json'
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']

        metadata_path = Path(data_dir) / 'metadata.json'
        metadata = json.load(open(metadata_path))['speakers']

        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances['feature_path'], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)

        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def get_speaker_number(self):
        return self.speaker_num


def collate_batch(batch):
    """对一个 batch 内的数据进行处理"""
    mel, speaker = zip(*batch)
    # 使用 pad_sequence 对 mel-spectrograms 进行填充
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    # 将 speaker 列表转换为一个 tensor
    speaker = torch.cat(speaker, dim=0)
    return mel, speaker


def get_dataloader(data_dir, batch_size, n_workers):
    """生成训练和验证用的 DataLoader"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]
    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=n_workers,
                              pin_memory=True,
                              collate_fn=collate_batch)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              num_workers=n_workers,
                              drop_last=False,  # 验证集通常不丢弃数据
                              pin_memory=True,
                              collate_fn=collate_batch)

    return train_loader, valid_loader, speaker_num


# ===================================================================
# Part 3: 高级模型架构 (Conformer + Self-Attention Pooling)
# ===================================================================
class Swish(nn.Module):
    """Swish 激活函数"""

    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvolutionModule(nn.Module):
    """Conformer 中的卷积模块"""

    def __init__(self, d_model, kernel_size=15, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1, padding=0)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=1,
                                        padding=(kernel_size - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x_residual + x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """单个 Conformer 模块"""

    def __init__(self, d_model, n_head, dim_feedforward, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.conv_module = ConvolutionModule(d_model, dropout=dropout)
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)
        x_residual = x
        x = x.transpose(0, 1)
        x_attn, _ = self.self_attn(x, x, x)
        x = x_residual + self.dropout(x_attn.transpose(0, 1))
        x = self.conv_module(x)
        x = x + 0.5 * self.ffn2(x)
        return self.layer_norm(x)


class AttentionPooling(nn.Module):
    """自注意力池化层 (Self-Attention Pooling)"""

    def __init__(self, d_model):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        attn_weights = torch.softmax(self.attention_net(x), dim=1)
        weighted_sum = torch.sum(x * attn_weights, dim=1)
        return weighted_sum


class Classifier(nn.Module):
    """整合了 Conformer 和 SAP 的分类器"""

    def __init__(self, input_dim=40, d_model=80, n_spks=600, n_head=4,
                 dim_feedforward=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(input_dim, d_model)

        self.conformer_blocks = nn.Sequential(
            *[ConformerBlock(d_model, n_head, dim_feedforward, dropout) for _ in range(num_layers)]
        )

        self.pooling = AttentionPooling(d_model)

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        out = self.prenet(mels)
        out = self.conformer_blocks(out)
        stats = self.pooling(out)
        out = self.pred_layer(stats)
        return out


# ===================================================================
# Part 4: 训练辅助函数
# ===================================================================
def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
                                    num_cycles: float = 0.5, last_epoch: int = -1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def model_fn(batch, model, criterion, device):
    mels, labels = batch
    mels = mels.to(device)
    labels = labels.to(device)

    outs = model(mels)

    loss = criterion(outs, labels)

    preds = outs.argmax(1)
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def valid(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0

    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc='Valid', unit='uttr')

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f'{running_loss / (i + 1):.4f}',
            accuracy=f'{running_accuracy / (i + 1):.4f}')

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


# ===================================================================
# Part 5: 训练主流程
# ===================================================================
def parse_args_train():
    config = {
        'data_dir': '/kaggle/input/ml2023springhw4/Dataset',
        'save_path': 'model.ckpt',
        'batch_size': 32,
        'n_workers': 2,  # Kaggle a环境通常建议用 2
        'valid_steps': 2000,
        'warmup_steps': 1000,
        'save_steps': 5000,
        'total_steps': 70000
    }
    return config


def main_train(data_dir, save_path, batch_size, n_workers, valid_steps, warmup_steps, total_steps, save_steps):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Info]: Use {device} now!')

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)

    print(f'[Info]: Finish loading data!', flush=True)

    # 使用升级后的 Conformer 模型
    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_accuracy = 0.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc='Train', unit=' step')

    for step in range(total_steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.4f}",
            accuracy=f"{batch_accuracy:.4f}",
            step=step + 1,
        )

        if (step + 1) % valid_steps == 0:
            pbar.close()
            valid_accuracy = valid(valid_loader, model, criterion, device)

            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()
                pbar.write(f"Step {step + 1}, new best model found. (accuracy={best_accuracy:.4f})")

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


# ===================================================================
# Part 6: 推理流程
# ===================================================================
class InferenceDataset(Dataset):
    def __init__(self, data_dir):
        testdata_path = Path(data_dir) / 'testdata.json'
        metadata = json.load(testdata_path.open())
        self.data_dir = data_dir
        self.data = metadata['utterances']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        utterance = self.data[index]
        feat_path = utterance['feature_path']
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        return feat_path, mel


# --- 已修复 ---
def inference_collate_batch(batch):
    """
    处理推理数据，同时返回原始长度以便去除填充
    """
    feat_paths, mels = zip(*batch)
    # 记录下每个 mel 的原始长度
    mel_lengths = [len(mel) for mel in mels]
    # 对 mel-spectrograms 进行填充
    padded_mels = pad_sequence([torch.FloatTensor(mel) for mel in mels], batch_first=True, padding_value=-20)
    return feat_paths, padded_mels, mel_lengths


def parse_args_inference():
    config = {
        'data_dir': '/kaggle/input/ml2023springhw4/Dataset',
        'model_path': 'model.ckpt',  # 使用刚刚训练保存的模型
        'output_path': 'submission.csv',
    }
    return config


# --- 已修复 ---
def main_inference(data_dir, model_path, output_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Info]: Use {device} for inference!')

    mapping_path = Path(data_dir) / 'mapping.json'
    mapping = json.load(mapping_path.open())
    speaker_num = len(mapping['id2speaker'])

    dataset = InferenceDataset(data_dir)
    dataloader = DataLoader(dataset,
                            batch_size=32,  # batch_size 可以根据你的显存大小调整
                            shuffle=False,
                            drop_last=False,
                            num_workers=2,
                            collate_fn=inference_collate_batch)  # <-- 使用修复后的 collate_fn

    model = Classifier(n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    segment_len = 128  # 使用和训练时差不多的长度

    results = [['Id', 'Category']]
    # 循环解包出原始长度
    for feat_paths, mels, mel_lengths in tqdm(dataloader, desc="Inference"):
        for i in range(len(feat_paths)):
            feat_path = feat_paths[i]
            original_len = mel_lengths[i]
            # 从填充后的 tensor 中，根据原始长度切片，得到无填充的 mel
            mel = mels[i, :original_len, :]

            # 如果 mel 太长，就进行分块投票
            if mel.shape[0] > segment_len:
                predictions = []
                # 使用滑动窗口
                for start in range(0, mel.shape[0], segment_len):
                    end = start + segment_len
                    if end > mel.shape[0]:
                        end = mel.shape[0]

                    # 跳过太短的片段，避免影响结果
                    if end - start < 10: continue

                    segment = mel[start:end].unsqueeze(0)  # 增加 batch 维度

                    with torch.no_grad():
                        segment = segment.to(device)
                        out = model(segment)
                        pred = out.argmax(1).cpu().item()
                        predictions.append(pred)

                # 投票选出最终预测结果
                if predictions:
                    final_pred = max(set(predictions), key=predictions.count)
                else:
                    # 如果没有任何有效的片段，则跳过
                    continue

            else:  # 如果 mel 本来就比较短
                with torch.no_grad():
                    mel = mel.unsqueeze(0).to(device)
                    out = model(mel)
                    final_pred = out.argmax(1).cpu().item()

            results.append([feat_path, mapping['id2speaker'][str(final_pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)

    print(f"[Info]: Inference complete. Submission saved to {output_path}")


# ===================================================================
# 启动入口
# ===================================================================
if __name__ == "__main__":
    # --- 训练阶段 ---
    print("=" * 20, "Starting Training", "=" * 20)
    train_config = parse_args_train()
    main_train(**train_config)
    print("=" * 20, "Training Finished", "=" * 20)

    # --- 推理阶段 ---
    # 训练结束后，自动使用保存的最佳模型进行推理
    print("\n" * 2)
    print("=" * 20, "Starting Inference", "=" * 20)
    inference_config = parse_args_inference()
    # 确保模型路径与训练时保存的路径一致
    inference_config['model_path'] = train_config['save_path']
    main_inference(**inference_config)
    print("=" * 20, "Inference Finished", "=" * 20)
