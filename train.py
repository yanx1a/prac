import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ========== 0. 设置随机种子，保证结果可复现 ==========
torch.manual_seed(42)
np.random.seed(42)

# 创建输出文件夹
os.makedirs("outputs", exist_ok=True)

# ========== 1. 加载数据 ==========
iris = load_iris()
X = iris.data.astype(np.float32)   # shape: (150, 4)
y = iris.target.astype(np.int64)   # shape: (150,)

# ========== 2. 先划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ========== 3. 标准化 ==========
# 注意：先用训练集 fit，再 transform 训练集和测试集，避免数据泄露
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

# ========== 4. 把 4 个特征重塑成 1×2×2 的“图像” ==========
# CNN 输入格式：[batch, channels, height, width]
X_train_cnn = X_train.reshape(-1, 1, 2, 2)
X_test_cnn = X_test.reshape(-1, 1, 2, 2)

# ========== 5. 转成 PyTorch 张量 ==========
X_train_t = torch.tensor(X_train_cnn, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_cnn, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# ========== 6. 定义 CNN 模型 ==========
class IrisCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：输入 1 通道，输出 8 个卷积核，核大小 2×2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=2)
        # 输入 2×2，卷积核 2×2，输出是 8×1×1
        self.fc1 = nn.Linear(8 * 1 * 1, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))      # 卷积 + ReLU
        x = x.view(x.size(0), -1)      # 展平
        x = F.relu(self.fc1(x))        # 全连接 + ReLU
        x = self.fc2(x)                # 输出 3 个类别的分数
        return x

model = IrisCNN()

# ========== 7. 损失函数和优化器 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ========== 8. 训练 ==========
epochs = 200
loss_history = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, y_train_t)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).argmax(dim=1)
            acc = (pred == y_test_t).float().mean().item()
        print(f"Epoch {epoch+1:3d} | Loss {loss.item():.4f} | Test Acc {acc:.4f}")

# ========== 9. 最终评估 ==========
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    pred = logits.argmax(dim=1).numpy()

print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=iris.target_names))

# 混淆矩阵
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()

# 损失曲线
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("outputs/loss_curve.png", dpi=200, bbox_inches="tight")
plt.close()

# 保存模型
torch.save(model.state_dict(), "outputs/iris_cnn.pth")
print("模型和图表已保存到 outputs/ 文件夹")