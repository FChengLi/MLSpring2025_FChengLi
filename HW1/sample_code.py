# ====== Imports ======
import math, os, csv, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# TensorBoard（可关）
from torch.utils.tensorboard import SummaryWriter

# Feature selection（对齐示例）
from sklearn.feature_selection import SelectKBest, f_regression
warnings.filterwarnings("ignore", category=UserWarning)

# ====== Reproducibility ======
def same_seed(seed: int):
    """设置随机种子，尽可能保证可复现。"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ====== Split (index-based) ======
def train_valid_split_np(arr: np.ndarray, valid_ratio: float, seed: int):
    """按行随机划分为 train/valid。"""
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
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx]) if self.y is not None else self.x[idx]

    def __len__(self):
        return self.x.shape[0]

# ====== Simple MLP（对齐示例容量；宽度由 config['layer'] 控制）======
class My_Model(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        out = self.layers(x)
        return out.squeeze(1)

# ====== Feature selection（对齐示例：KBest）======
def select_feat(train_data, valid_data, test_data, use_kbest=True, k=16):
    """
    返回 (x_train, x_valid, x_test, y_train, y_valid)。
    - 约定标签在最后一列。
    - use_kbest=False 时使用全部特征；True 时使用 SelectKBest(f_regression, k)
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if not use_kbest:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(raw_x_train, y_train)
        idx = np.argsort(selector.scores_)[::-1]
        feat_idx = list(np.sort(idx[:k]))

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

# ====== Train one model（MSE + SGD/Adam；对齐示例）======
def trainer(train_loader, valid_loader, model, config, device, valid_scores_ref):
    criterion = nn.MSELoss(reduction='mean')

    if config['optim'] == 'SGD':
        if config['no_momentum']:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=config['learning_rate'],
                                        weight_decay=config['weight_decay'])
        else:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=config['learning_rate'],
                                        momentum=config['momentum'],
                                        weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])

    writer = None if config['no_tensorboard'] else SummaryWriter()
    os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)

    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    for epoch in range(1, n_epochs + 1):
        # ---- train ----
        model.train()
        loss_record = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())
        mean_train_loss = float(np.mean(loss_record))

        # ---- validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_losses.append(criterion(pred, y).item())
        mean_valid_loss = float(np.mean(val_losses))

        if writer:
            writer.add_scalar('Loss/train', mean_train_loss, step)
            writer.add_scalar('Loss/valid', mean_valid_loss, step)

        # 保存策略与示例一致：仅当当前折是最优时覆盖
        if mean_valid_loss < best_loss - 1e-12:
            best_loss = mean_valid_loss
            if len(valid_scores_ref):
                if best_loss < min(valid_scores_ref):
                    torch.save(model.state_dict(), config['save_path'])
                    print(f'  ↳ Saved model (val={best_loss:.4f})')
            else:
                torch.save(model.state_dict(), config['save_path'])
                print(f'  ↳ Saved model (val={best_loss:.4f})')
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= config['early_stop']:
                print(f'Best val {best_loss:.4f}. Early stop.')
                break

    if writer:
        writer.close()
    return best_loss

# ====== Predict ======
@torch.no_grad()
def predict(test_loader, model, device):
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        pred = model(x)
        preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy()

# ====== Save predictions ======
def save_pred(preds, file):
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, float(p)])

# ====== Objective（保留示例风格：K 折 + 可选标准化 + KBest）======
def objective():
    global training_data, test_data, valid_scores, config

    print(f'''hyper-parameter: 
        optimizer: {config['optim']},
        lr: {config['learning_rate']}, 
        batch_size: {config['batch_size']}, 
        k: {config['k']}, 
        layer: {config['layer']}''')

    valid_scores = []
    kfold = int(1 / config['valid_ratio'])
    num_valid_samples = len(training_data) // kfold

    # 打乱一次，和示例保持一致的行为
    rng = np.random.default_rng(config['seed'])
    shuffled = training_data[rng.permutation(len(training_data))]

    for fold in range(kfold):
        # Data split
        valid_data = shuffled[num_valid_samples * fold : num_valid_samples * (fold + 1)]
        train_data = np.concatenate((
            shuffled[:num_valid_samples * fold],
            shuffled[num_valid_samples * (fold + 1):]
        ), axis=0)

        # Normalization（默认关闭，对齐示例：no_normal=True）
        if not config['no_normal']:
            train_mean = np.mean(train_data[:, 35:-1], axis=0)
            train_std  = np.std(train_data[:, 35:-1], axis=0) + 1e-8
            train_data[:, 35:-1] = (train_data[:, 35:-1] - train_mean) / train_std
            valid_data[:, 35:-1] = (valid_data[:, 35:-1] - train_mean) / train_std
            test_data[:,  35:  ] = (test_data[:,  35:  ] - train_mean) / train_std

        # Feature selection（KBest）
        x_train, x_valid, x_test, y_train, y_valid = select_feat(
            train_data, valid_data, test_data,
            use_kbest=config['no_select_all'], k=config['k']
        )

        # Datasets
        train_ds = COVID19Dataset(x_train, y_train)
        valid_ds = COVID19Dataset(x_valid, y_valid)
        test_ds  = COVID19Dataset(x_test)

        # DataLoaders（对齐示例：valid_loader 也 shuffle=True）
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=True,  pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'], shuffle=False, pin_memory=True)

        # Model（与示例容量对齐）
        h1, h2 = config['layer'][0], config['layer'][1]
        model = My_Model(input_dim=x_train.shape[1], hidden1=h1, hidden2=h2).to(device)

        # Train one fold
        best_val = trainer(train_loader, valid_loader, model, config, device, valid_scores)
        valid_scores.append(best_val)

        # 示例代码里的行为：如果配置了不做K折（no_k_cross=False 表示“要做K折”），则这里 break。
        if not config['no_k_cross']:
            break

        # 欠拟合快速跳出（和示例一致的小策略）
        if best_val > 2:
            print(f'在第{fold+1}折上欠拟合，提前结束该次训练')
            break

    print(f'valid_scores: {valid_scores}')
    return x_test, test_loader

# ====== Main ======
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # —— 对齐示例的默认配置（关键差异：仍保留我们框架的函数/类名）——
    config = {
        'seed': 5201314,
        'k': 16,                 # KBest 选择的特征数（核心！）
        'layer': [16, 16],       # 模型宽度（与示例一致）
        'optim': 'SGD',          # 'SGD' or 'Adam'
        'momentum': 0.85,
        'valid_ratio': 0.05,
        'n_epochs': 10000,
        'batch_size': 256,
        'learning_rate': 1e-5,
        'weight_decay': 1e-5,
        'early_stop': 600,
        'save_path': './models/model.ckpt',
        'no_select_all': True,   # True -> 使用 KBest（对齐示例含义）
        'no_momentum': False,     # True -> 不启用 momentum
        'no_normal': True,       # True -> 不做标准化（对齐示例默认）
        'no_k_cross': False,     # False -> 执行 K 折（对齐示例行为）
        'no_tensorboard': True, # 是否写入 TensorBoard
    }

    # reproducibility
    same_seed(config['seed'])

    # read data
    training_data = pd.read_csv('./covid_train.csv').values
    test_data     = pd.read_csv('./covid_test.csv').values

    # run (单次；可在外层再做多种子/多次集成)
    print('开始训练（对齐示例的最小改动版）...')
    x_test, test_loader = objective()

    # 推断与保存（加载已保存的最优权重）
    model = My_Model(input_dim=x_test.shape[1],
                     hidden1=config['layer'][0],
                     hidden2=config['layer'][1]).to(device)
    model.load_state_dict(torch.load(config['save_path'], map_location=device))
    preds = predict(test_loader, model, device)
    save_pred(preds, 'submission.csv')
    print('Saved to submission.csv')
