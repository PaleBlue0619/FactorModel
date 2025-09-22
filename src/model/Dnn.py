import torch
import torch.nn as nn
import torch.nn.functional as func
from skorch import NeuralNetRegressor
import warnings
warnings.filterwarnings("ignore")

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2):
        super(DNN, self).__init__()
        self.criterion = nn.MSELoss()  # 定义损失函数
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加 BatchNorm
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)  # 添加 BatchNorm
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)   # 回归任务不需要激活函数
        self.dropout = nn.Dropout(dropout)
        # self.shortcut = nn.Linear(input_dim, output_dim)  # 残差连接

    def forward(self, X):
        out = F.relu(self.fc1(X))
        out = self.dropout(out)
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        # shortcut = self.shortcut(X)             # 残差连接
        return out # + shortcut

    def compute_loss(self, X, y):
        output = self(X)                        # 前向传播，获取模型预测值
        loss = self.criterion(output, y)        # 计算损失值
        return loss

    def train_step(self, batch, *args, **kwargs):
        X, y = batch
        loss = self.compute_loss(X, y)          # 计算损失值
        self.optimizer.zero_grad()              # 梯度清零
        loss.backward()                         # 反向传播
        self.optimizer.step()                   # 参数更新
        return loss.item()

# Wrapper
class CustomDNN(DNN):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2):
        super().__init__(input_dim, hidden_dim, output_dim, dropout)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # 定义优化器
        self.criterion = nn.MSELoss()

    # 重写train_step方法
    # @overrides
    def train_step(self, batch, *args, **kwargs):
        X, y = batch
        loss = self.compute_loss(X, y)          # 计算损失值
        self.optimizer.zero_grad()              # 梯度清零
        loss.backward()                         # 反向传播
        self.optimizer.step()                   # 参数更新
        return loss.item()


# Get Method
def get_DNN(input_dim: int, **kwargs):
    model = CustomDNN(input_dim)
    DNN_net = NeuralNetRegressor(
        module=model,
        module__input_dim=input_dim,
        module__output_dim=1,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    )
    return DNN_net.initialize()