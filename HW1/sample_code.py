# ====== Imports ======
import math, os, csv
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# ====== Reproducibility ======
def same_seed(seed: int):
    """设置随机种子，尽可能保证可复现。"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 可选：更强确定性（可能降低速度/报不支持的算子）
    # torch.use_deterministic_algorithms(True)

# ====== Split (index-based, zero-copy) ======
def train_valid_split_np(arr: np.ndarray, valid_ratio: float, seed: int):
    """
    对 numpy 数组按行进行随机划分，返回两个 numpy 视图（通过索引切分，无多余拷贝）。
    arr: (N, D) 训练原始矩阵
    """
    N = len(arr)
    n_valid = int(round(valid_ratio * N))
    rng = np.random.default_rng(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    valid_idx = indices[:n_valid]
    train_idx = indices[n_valid:]
    return arr[train_idx], arr[valid_idx]

# ====== Dataset ======
class COVID19Dataset(Dataset):
    """
    x: np.ndarray or torch.FloatTensor, shape (N, F)
    y: np.ndarray or torch.FloatTensor, shape (N,) or None
    """
    def __init__(self, x, y=None):
        x = torch.as_tensor(x, dtype=torch.float32)
        self.x = x
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

    def __len__(self):
        return self.x.shape[0]

# ====== Simple MLP ======
class My_Model(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # 稍加宽并加入轻微 Dropout 与 BatchNorm 提升稳定性
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = self.layers(x)            # (B, 1)
        return out.squeeze(1)           # -> (B,)

# ====== Feature selection ======
def select_feat(train_data, valid_data, test_data, select_all=True, keep_idx=None):
    """
    选择用于回归的特征列并返回 (x_train, x_valid, x_test, y_train, y_valid)。
    - 约定标签在最后一列。
    - keep_idx: 显式保留的特征列索引列表（相对于 raw_x_*）。
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    elif keep_idx is not None:
        feat_idx = list(keep_idx)
    else:
        # 示例：跳过前 35 列 state one-hot，只用行为/症状等连续特征
        feat_idx = list(range(35, raw_x_train.shape[1]))

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

# ====== Standardize ======
class Standardizer:
    """简单的标准化器：(x - mean) / std（避免使用外部库，便于比赛打包）"""
    def fit(self, x: np.ndarray):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_  = x.std(axis=0, keepdims=True) + 1e-8
        return self
    def transform(self, x: np.ndarray):
        return (x - self.mean_) / self.std_
    def fit_transform(self, x: np.ndarray):
        return self.fit(x).transform(x)

# ====== Train one model ======
def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')              # 任务指定 MSE
    optimizer = torch.optim.AdamW(                        # 更稳的默认选择
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=False
    )

    writer = SummaryWriter(log_dir=config.get('log_dir', None))

    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)

    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        loss_record = []
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 防爆梯
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
        mean_train_loss = float(np.mean(loss_record))
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # ---- validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())
        mean_valid_loss = float(np.mean(val_losses))
        rmse = math.sqrt(mean_valid_loss)
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('Metric/RMSE', rmse, step)
        print(f'Epoch [{epoch}/{n_epochs}] '
              f'Train MSE: {mean_train_loss:.6f} | Valid MSE: {mean_valid_loss:.6f} (RMSE {rmse:.5f})')

        scheduler.step(mean_valid_loss)

        # ---- save & early stop ----
        if mean_valid_loss < best_loss - 1e-7:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            early_stop_count = 0
            print(f'  ↳ Saved best model (MSE={best_loss:.6f})')
        else:
            early_stop_count += 1
            if early_stop_count >= config['early_stop']:
                print('Early stopping: validation did not improve.')
                break

# ====== Inference ======
@torch.no_grad()
def predict(test_loader, model, device):
    """评估模式下批量推断，返回 numpy 数组 shape (N,)."""
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        pred = model(x)
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy()

def save_pred(preds, file):
    """按 Kaggle 要求保存预测结果。"""
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, float(p)])
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 5201314,
        'select_all': True,
        'valid_ratio': 0.2,
        'n_epochs': 2000,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'early_stop': 200,
        'save_path': './models/model.ckpt',
        'log_dir': './runs/hw1_baseline_adamw'
    }

    same_seed(config['seed'])

    # 1) 读取数据（文件名与你目录一致）
    # 强健读取：自动嗅探分隔符，兼容常见编码
    train_data = pd.read_csv(
        r'E:\MLSpring2025_HW\HW1\covid_train.csv',  # 用绝对路径更稳
        engine='python',  # 允许自动嗅探分隔符
        sep=None,  # 让 pandas 用 Sniffer 检测 , ; \t 等
        encoding='utf-8-sig'  # 兼容带 BOM 的 utf-8
    ).values

    test_data = pd.read_csv(
        r'E:\MLSpring2025_HW\HW1\covid_test.csv',
        engine='python',
        sep=None,
        encoding='utf-8-sig'
    ).values

    # 2) 训练/验证划分
    train_arr, valid_arr = train_valid_split_np(train_data, config['valid_ratio'], config['seed'])
    print(f"train_data size: {train_arr.shape}\nvalid_data size: {valid_arr.shape}\ntest_data size: {test_data.shape}")

    # 3) 选特征
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_arr, valid_arr, test_data, config['select_all'])

    # 4) 标准化（fit 在训练集，transform 在验证/测试）
    scaler = Standardizer()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test  = scaler.transform(x_test)

    print(f'number of features: {x_train.shape[1]}')

    # 5) DataLoader
    train_ds = COVID19Dataset(x_train, y_train)
    valid_ds = COVID19Dataset(x_valid, y_valid)
    test_ds  = COVID19Dataset(x_test)

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # 6) 训练
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    trainer(train_loader, valid_loader, model, config, device)

    # 7) 推断（加载最佳权重）
    best_model = My_Model(input_dim=x_train.shape[1]).to(device)
    best_model.load_state_dict(torch.load(config['save_path'], map_location=device))
    preds = predict(test_loader, best_model, device)

    # 8) 保存提交文件
    save_pred(preds, 'pred.csv')
    print('Saved to pred.csv')


