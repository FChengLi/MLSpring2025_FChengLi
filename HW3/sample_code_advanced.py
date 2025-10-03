import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" 和 "Subset" 在进行半监督学习时可能会很有用。
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# 导入预训练模型
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 这是用于显示进度条的库。
from tqdm import tqdm
import random

# 用于 t-SNE 可视化
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %% md
# 为确保可复现性，固定随机种子
myseed = 6666
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
# %% md
### 数据增强 (Transforms)
# 像 ResNet 这样的预训练模型通常是在 224x224 分辨率的图像上训练的。
# 我们还需要使用 ImageNet 数据集的均值和标准差来对图像进行标准化。
# %% md
# 对于测试集，我们只需要调整大小和标准化
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 对于训练集，我们应用更全面的数据增强来使模型更具鲁棒性。
# 每次对同一张图片调用此转换时，都会产生不同的结果。
train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    # 添加随机增强
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转图像
    transforms.RandomRotation(25),  # 随机旋转图像最多25度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机改变颜色属性
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机平移和缩放

    # ToTensor() 操作应在标准化之前应用。
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# %% md
### 数据集
# %% md
class FoodDataset(Dataset):

    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files

        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname).convert('RGB')  # 确保图像是RGB格式
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1  # 测试集没有标签

        return im, label


# %% md
### 模型 (使用预训练的 ResNet-50)
# %% md
# 我们将使用 torchvision 中一个预训练好的 ResNet-50 模型。
# 'weights=models.ResNet50_Weights.DEFAULT' 会加载当前可用的最佳预训练权重。
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# 原始的 ResNet-50 是在含有1000个类别的 ImageNet 上训练的。
# 我们需要替换其最终的全连接层，以匹配我们的类别数量（11类）。
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 11)

# 为了加速训练，你可以冻结早期的层，只训练后面的层。
# 例如，冻结除最后一层外的所有层：
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.fc.parameters():
#     param.requires_grad = True

# %% md
### 参数配置
# %% md
# 只有当 GPU 可用时才使用 "cuda"。
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备: {device}")

# 将模型移动到指定设备。
model = model.to(device)

# 批次大小。如果遇到内存不足 (out of memory) 的问题，可以减小此值。
batch_size = 32

# 增加训练周期数以获得更好的收敛效果。
n_epochs = 30

# 如果连续 'patience' 个周期性能没有提升，则提前停止。
patience = 5

# 对于分类任务，我们使用交叉熵作为损失函数。
criterion = nn.CrossEntropyLoss()

# 我们可以为预训练层和新层设置不同的学习率。
# 不过，对所有参数使用一个优化器进行微调通常效果也很好。
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# 添加一个学习率调度器，当验证集损失停滞时降低学习率。
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# %% md
### 数据加载器 (Dataloader)
# %% md
# 构建训练集和验证集。
train_set = FoodDataset("/kaggle/input/ml2023spring-hw3/train", tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_set = FoodDataset("/kaggle/input/ml2023spring-hw3/valid", tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
# %% md
### 开始训练
# %% md
# 初始化追踪器，这些不是模型的参数，不应被更改。
stale = 0
best_acc = 0
_exp_name = "resnet_finetune" # 定义一个实验名称

for epoch in range(n_epochs):
    # ---------- 训练阶段 ----------
    model.train()
    train_loss = []
    train_accs = []
    for batch in tqdm(train_loader, desc=f"第 {epoch + 1}/{n_epochs} 周期 [训练]"):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ 训练 | {epoch + 1:03d}/{n_epochs:03d} ] 损失 = {train_loss:.5f}, 准确率 = {train_acc:.5f}")

    # ---------- 验证阶段 ----------
    model.eval()
    valid_loss = []
    valid_accs = []
    for batch in tqdm(valid_loader, desc=f"第 {epoch + 1}/{n_epochs} 周期 [验证]"):
        imgs, labels = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"[ 验证 | {epoch + 1:03d}/{n_epochs:03d} ] 损失 = {valid_loss:.5f}, 准确率 = {valid_acc:.5f}")

    # 更新学习率
    scheduler.step(valid_loss)

    # 保存最佳模型
    if valid_acc > best_acc:
        print(f"在第 {epoch + 1} 个周期发现最佳模型，正在保存... 准确率: {valid_acc:.5f}")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt")
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale >= patience:
            print(f"连续 {patience} 个周期性能未提升，提前停止训练。")
            break
# %% md
### 测试集的数据加载器
# %% md
test_set = FoodDataset("/kaggle/input/ml2023spring-hw3/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
# %% md
### 进行测试并生成预测的 CSV 文件
# %% md
# 为测试创建一个新的模型实例
model_best = models.resnet50(weights=None)  # 不加载预训练权重，我们将加载自己微调过的权重
num_ftrs = model_best.fc.in_features
model_best.fc = nn.Linear(num_ftrs, 11)
model_best = model_best.to(device)

# 加载保存的最佳权重
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data, _ in tqdm(test_loader, desc="测试中"):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


# %% md
# 创建提交所需的 CSV 文件
def pad4(i):
    return "0" * (4 - len(str(i))) + str(i)


df = pd.DataFrame()
# 测试集是按文件名排序的，所以我们可以直接从 0 到 len(test_set)-1 生成 ID
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv", index=False)
print("提交文件 (submission.csv) 已创建！")


# %% md
### Q2. t-SNE 可视化
# 这一部分会从训练好的模型中提取特征并进行可视化。
# 请在模型完全训练好并且最佳检查点（checkpoint）已保存后运行此部分。
# %% md
def extract_features(model, dataloader, layer):
    """
    从模型的指定层提取特征。
    'layer' 参数可以是 'mid' (中层) 或 'top' (顶层)。
    """
    features = []
    labels_list = []

    # 使用一个钩子 (hook) 来捕获所需层的输出
    feature_output = None

    def hook(module, input, output):
        nonlocal feature_output
        feature_output = output

    handle = None
    if layer == 'mid':
        # ResNet 中的 'layer3' 是一个很好的中层特征层
        handle = model.layer3.register_forward_hook(hook)
    elif layer == 'top':
        # 'avgpool' 是分类器前的最后一层，代表了高层特征
        handle = model.avgpool.register_forward_hook(hook)
    else:
        raise ValueError("Layer 参数必须是 'mid' 或 'top'")

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc=f"正在提取 {layer} 层特征"):
            _ = model(imgs.to(device))
            # 钩子现在已经捕获了输出

            # 对中层特征的空间维度进行池化
            if layer == 'mid':
                # 使用自适应平均池化将空间维度降到 1x1
                pooled_output = nn.AdaptiveAvgPool2d((1, 1))(feature_output)
            else:  # layer == 'top'
                pooled_output = feature_output

            # 展平并移动到 CPU
            flattened_output = pooled_output.view(pooled_output.size(0), -1).cpu().numpy()
            features.append(flattened_output)
            labels_list.append(labels.numpy())

    # 移除钩子
    handle.remove()

    return np.concatenate(features), np.concatenate(labels_list)


def plot_tsne(features, labels, title, filename):
    """
    执行 t-SNE 并绘制结果。
    """
    print(f"正在为“{title}”执行 t-SNE...")
    tsne = TSNE(n_components=2, random_state=myseed, perplexity=30, n_iter=1000)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(11)), title="类别")
    plt.title(title)
    plt.xlabel("t-SNE 特征维度 1")
    plt.ylabel("t-SNE 特征维度 2")
    plt.savefig(filename)
    plt.show()
    print(f"t-SNE 图像已保存为 {filename}")


# --- 运行可视化 ---
# 确保最佳模型已加载
print("正在为可视化加载最佳模型...")
model_best = models.resnet50(weights=None)
num_ftrs = model_best.fc.in_features
model_best.fc = nn.Linear(num_ftrs, 11)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best = model_best.to(device)

# 使用验证集进行可视化
valid_loader_viz = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 1. 中层特征
mid_features, mid_labels = extract_features(model_best, valid_loader_viz, 'mid')
plot_tsne(mid_features, mid_labels, '中层特征 (来自 layer3) 的 t-SNE 可视化', 'tsne_mid.png')

# 2. 顶层特征
top_features, top_labels = extract_features(model_best, valid_loader_viz, 'top')
plot_tsne(top_features, top_labels, '顶层特征 (来自 avgpool) 的 t-SNE 可视化', 'tsne_top.png')