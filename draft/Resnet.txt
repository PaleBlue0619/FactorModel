import torch
import torch.nn as nn
import torch.nn.functional as func
from skorch import NeuralNetRegressor
import warnings
warnings.filterwarnings("ignore")


# 定义残差块（全连接版本）
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 如果输入和输出的维度不一致，使用1x1线性层调整维度
        self.shortcut = nn.Sequential()
        if input_dim != hidden_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)  # 跳跃连接
        out = F.relu(out)
        return out

# 定义ResNet（全连接版本）
class ResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.2, num_blocks=None):
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        super(ResNet, self).__init__()
        self.criterion = nn.MSELoss()  # 定义损失函数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 初始全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # 残差块
        self.layer1 = self._make_layer(hidden_dim, num_blocks[0], dropout)
        self.layer2 = self._make_layer(hidden_dim * 2, num_blocks[1], dropout)
        self.layer3 = self._make_layer(hidden_dim * 4, num_blocks[2], dropout)
        self.layer4 = self._make_layer(hidden_dim * 8, num_blocks[3], dropout)

        # 全连接层
        self.fc2 = nn.Linear(hidden_dim * 8, output_dim)

    def _make_layer(self, hidden_dim, num_blocks, dropout):
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(self.input_dim, hidden_dim, dropout))
            self.input_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc2(x)
        return x

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
class CustomResNet(ResNet):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1,dropout=0.2, num_blocks=None):
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        super().__init__(input_dim=input_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         dropout=dropout,
                         num_blocks=num_blocks)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # 定义优化器
        self.criterion = nn.MSELoss()

    def train_step(self, batch, *args, **kwargs):
        X, y = batch
        loss = self.compute_loss(X, y)          # 计算损失值
        self.optimizer.zero_grad()              # 梯度清零
        loss.backward()                         # 反向传播
        self.optimizer.step()                   # 参数更新
        return loss.item()

# Get Method
def get_RESNET(input_dim: int, **kwargs):
    model = CustomResNet(input_dim)
    RESNET_net = NeuralNetRegressor(
        module=model,
        module__input_dim=input_dim,
        module__output_dim=1,
        criterion=nn.MSELoss,
        optimizer=torch.optim.Adam,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    )
    return RESNET_net.initialize()