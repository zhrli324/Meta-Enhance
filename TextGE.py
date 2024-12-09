import torch
from torch import nn, tensor
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GroupEnhance(nn.Module):
    def __init__(self, hyper_training, hyper_model, groups=5):
        super(GroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化，获取每个 group 的代表特征向量
        self.weight = Parameter(torch.zeros(1, groups, 1))
        self.bias = Parameter(torch.ones(1, groups, 1))
        self.sig = nn.Sigmoid()
        self.hyper_training = hyper_training
        self.hyper_model = hyper_model

    def forward(self, x):  # (B, f, l)
        """
        :param x: 上层输出的特征图 (B, f, l)
        :return: 增强后的特征图，(B, f, l)
        """
        b, f, l = x.size()  # e.g., (2,20,10)
        # 确保 f 能被 groups 整除
        assert f % self.groups == 0, f"Feature dimension f={f} must be divisible by groups={self.groups}"
        x = x.view(b * self.groups, f // self.groups, l)  # (B*groups, f', l)
        if self.hyper_training:
            input_features = x.size(1)
            # 建议在 __init__ 中初始化 hyper_model 的子模块，而不是在 forward 中
            # 避免每次前向传播时重复初始化
            # 这里假设 hyper_model 已经被正确初始化
            k = self.hyper_model(x)  # (B*groups, f', l)
            k = k.permute(0, 2, 1)  # (B*groups, l, f')
            dot = x * k  # (B*groups, f', l)
        else:
            dot = x * self.avg_pool(x)  # 平均值与每个位置进行点积 (B*groups, f', l) * (B*groups, f', 1) = (B*groups, f', l)
        dot = dot.sum(dim=1, keepdim=True)  # 获得每个 group 的系数图 (B*groups, 1, l)
        norm = dot.view(b * self.groups, -1)  # (B*groups, l)
        norm = norm - norm.mean(dim=1, keepdim=True)  # 除了 batch * groups 外，计算均值
        std = norm.std(dim=1, keepdim=True) + 1e-5  # 标准差 (B*groups, 1)
        norm = norm / std  # 标准化，数据在 0 附近分布 (B*groups, l)
        norm = norm.view(b, self.groups, l)  # (B, groups, l)
        norm = norm.to(device)
        norm = norm * self.weight + self.bias  # (B, groups, l)
        norm = norm.view(b * self.groups, 1, l)  # (B*groups, 1, l)
        x = x * self.sig(norm)  # (B*groups, f', l)
        x = x.view(b, f, l)  # (B, f, l)
        return x
