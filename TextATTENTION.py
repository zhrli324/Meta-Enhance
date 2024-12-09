import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    基于加性（Additive）注意力机制的模块
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)  # 确保这里的 hidden_dim 与 hidden_states 的最后一维一致
        self.context_vector = nn.Parameter(torch.randn(hidden_dim))  # (H,)

    def forward(self, hidden_states):
        """
        :param hidden_states: (B, L, H), 输入的隐藏状态，B 是批量大小，L 是序列长度，H 是隐藏状态维度
        :return: (B, H) 加权平均后的输出
        """
        # Step 1: 计算每个时间步的注意力得分
        scores = torch.tanh(self.attn(hidden_states))  # (B, L, H)
        
        # Step 2: 计算注意力得分与 context_vector 的点积，注意需要进行维度匹配
        scores = torch.matmul(scores, self.context_vector.unsqueeze(-1))  # (B, L, 1)
        scores = scores.squeeze(-1)  # (B, L)

        # Step 3: 计算注意力权重（注意力得分经过 softmax）
        attn_weights = torch.softmax(scores, dim=1)  # (B, L)

        # Step 4: 对所有的隐藏状态加权求和
        weighted_sum = torch.bmm(attn_weights.unsqueeze(1), hidden_states)  # (B, 1, H)
        weighted_sum = weighted_sum.squeeze(1)  # (B, H)

        return weighted_sum, attn_weights


class TextRNNWithAttention(nn.Module):
    """带有全局注意力机制的 RNN"""
    def __init__(self, cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate):
        """
        :param cell: 隐藏单元类型 'rnn', 'bi-rnn', 'gru', 'bi-gru', 'lstm', 'bi-lstm'
        :param vocab_size: 词表大小
        :param embed_size: 词嵌入维度
        :param hidden_dim: 隐藏神经元数量
        :param num_layers: 隐藏层数
        :param dropout_rate: dropout 率
        """
        super(TextRNNWithAttention, self).__init__()
        self._cell = cell

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = None
        if cell == 'rnn':
            self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
            out_hidden_dim = hidden_dim
        elif cell == 'bi-rnn':
            self.rnn = nn.RNN(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim
        elif cell == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
            out_hidden_dim = hidden_dim
        elif cell == 'bi-gru':
            self.rnn = nn.GRU(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim
        elif cell == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
            out_hidden_dim = hidden_dim
        elif cell == 'bi-lstm':
            self.rnn = nn.LSTM(embed_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
            out_hidden_dim = 2 * hidden_dim
        else:
            raise Exception("no such rnn cell")

        # 初始化注意力机制
        self.attention = Attention(out_hidden_dim)  # 确保 hidden_dim 与 RNN 输出匹配

        self.dropout = nn.Dropout(dropout_rate)
        # 移除输出层
        # self.output_layer = nn.Linear(out_hidden_dim, 1)  # 假设做分类任务

    def forward(self, x):
        """
        :param x: 输入的文本序列 (B, L)
        :return: 加权求和后的注意力输出 (B, f, l)
        """
        x = x.long()
        embedded = self.embedding(x)  # (B, L, E)
        
        # 通过 RNN 获取每个时间步的隐藏状态
        rnn_out, _ = self.rnn(embedded)  # (B, L, H)

        # 通过注意力机制获得加权输出
        attn_output, attn_weights = self.attention(rnn_out)  # (B, H), (B, L)

        # Dropout
        attn_output = self.dropout(attn_output)  # (B, H)

        # 重新调整形状为 (B, f, l)，例如 f=20, l=10
        f = 100
        l = attn_output.size(1) // f  # 确保 H = f * l
        attn_output = attn_output.view(x.size(0), f, l)

        return attn_output, attn_weights  # 返回特征图



class TextCNNWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, filter_num, filter_sizes, dropout_rate):
        super(TextCNNWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, filter_num, (filter_size, embed_size)) for filter_size in filter_sizes
        ])
        
        # 修正后的 Attention 模块的 hidden_dim
        self.attention = Attention(filter_num * len(filter_sizes))  # 现在 hidden_dim = 200

        self.dropout = nn.Dropout(dropout_rate)
        # 移除输出层
        # self.output_layer = nn.Linear(len(filter_sizes) * filter_num, 1)  # 输出层

    def forward(self, x):
        """
        :param x: 输入的文本序列 (B, L)
        :return: 加权求和后的注意力输出 (B, f, l)
        """
        x = x.long()
        embedded = self.embedding(x)  # (B, L, E)
        embedded = embedded.unsqueeze(1)  # (B, 1, L, E) 这里加一个维度为1，表示通道数

        # 每个卷积层对应一个特定的 filter_size
        conv_outs = [F.relu(conv(embedded)).squeeze(3) for conv in self.conv_layers]  # (B, filter_num, L-filter_size+1)
        pooled_outs = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outs]  # (B, filter_num)

        # 连接所有卷积层的输出
        cnn_out = torch.cat(pooled_outs, dim=1)  # (B, filter_num * len(filter_sizes)) = (B, 200)

        # 通过注意力机制
        attn_output, _ = self.attention(cnn_out.unsqueeze(1))  # 注意力的输入是 (B, 1, 200)

        # Dropout
        attn_output = self.dropout(attn_output)  # (B, 1, 200)

        # 重新调整形状为 (B, f, l)，例如 f=20, l=10
        f = 100
        l = 2
        attn_output = attn_output.view(x.size(0), f, l) 

        return attn_output  # (B, f, l)
