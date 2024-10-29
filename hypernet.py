import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, input_features, lstm_hidden_size, output_size):
        """
        使用 LSTM 处理变长输入序列并输出一个固定维度的张量。

        参数:
        - input_features: 每个数据的特征维度 f。
        - lstm_hidden_size: LSTM 隐状态的大小。
        - output_size: 输出的维度大小。
        """
        super(LSTMModel, self).__init__()

        # 定义 LSTM 层
        self.lstm = nn.LSTM(input_features, lstm_hidden_size, batch_first=True)

        # 输出层，用来将 LSTM 的输出映射到目标输出维度
        self.output_layer = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        """
        前向传播，输入形状为 (b*groups, -1, l)，输出形状为 (b*groups, -1, 1)。
        """
        # 获取输入的形状
        b_groups, seq_len, f = x.shape  # (b*groups, l, f)

        # 将输入维度 (b*groups, seq_len, f) 转换为 LSTM 期望的输入格式 (b*groups, f, seq_len)
        x = x.permute(0, 2, 1)  # 交换维度，变成 (b*groups, f, seq_len)

        # 通过 LSTM 层，LSTM 输出 (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 我们使用 LSTM 的输出去计算输出层
        # 这里 lstm_out 的形状为 (b*groups, seq_len, lstm_hidden_size)
        output = self.output_layer(lstm_out)  # 形状为 (b*groups, seq_len, output_size)

        # 最后输出形状为 (b*groups, seq_len, 1)
        return output