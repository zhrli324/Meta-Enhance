import torch
from torch import nn, tensor
from torch.nn.parameter import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GroupEnhance(nn.Module):
    def __init__(self, hyper_training, hyper_model, groups = 5):
        super(GroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool1d(1)# 自适应平均池化，获取每个group的代表特征向量
        self.weight   = Parameter(torch.zeros(1, groups, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1))
        self.sig      = nn.Sigmoid()
        self.hyper_training = hyper_training
        self.hyper_model = hyper_model

    def forward(self, x): # (b, f, l)=(2,20,2)=80 group = 4
        """
        :param x: 上层输出的特征map (b, f, l)
        :return: 增强后的特征map，(b, f, l)
        """
        b, f, l = x.size() # b=l=2,f=20
        x = x.view(b*self.groups, -1, l) # (2*4,5,2)
        if self.hyper_training:
            input_features = len(x[-1])
            self.hyper_model.lstm = nn.LSTM(input_features, 32, 1, batch_first=True)
            self.hyper_model.output_layer = nn.Linear(32, input_features)
            self.hyper_model.to(device)
            k = self.hyper_model(x)
            k = k.permute(0, 2, 1)
            dot = x * k
        else:
            dot = x * self.avg_pool(x) # 平均值与每个位置进行点积(8,5,2)*(8*5*1)=(8,5,2)
        dot = dot.sum(dim=1, keepdim=True) # 获得每个group的系数map (8,1,2)
        norm = dot.view(b * self.groups, -1) # (8,2)
        norm = norm - norm.mean(dim=1, keepdim=True) # 除了batch * groups外，计算均值
        std = norm.std(dim=1, keepdim=True) + 1e-5 # 标准差(2,1)
        norm = norm / std #标准化，是数据在0左右分布 (8,2)
        norm = norm.view(b, self.groups, l) #(2,4,2)
        norm.to(device)
        self.to(device)
        norm = norm * self.weight + self.bias
        norm = norm.view(b * self.groups, 1, l)
        x = x * self.sig(norm) #(b*group,f/group,l)(2*4,5,2)
        x = x.view(b, f, l)
        return x
