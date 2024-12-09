import torch
from torch import nn
import TextATTENTION as a  # 导入注意力机制模块
import TextGE as e  # GroupEnhance 模块
import torch.nn.functional as F

class ge(nn.Module):
    def __init__(self, cell, vocab_size, embed_size, filter_num, filter_size, hidden_dim, num_layers, class_num,
                 dropout_rate, g, k, hyper_training=False, hyper_model=None):
        super(ge, self).__init__()
        self.hyper_model = hyper_model
        self.hyper_training = hyper_training
        
        # 使用 TextATTENTION 替换原来的 TextCNN 和 TextRNN
        self.attention_cnn = a.TextCNNWithAttention(vocab_size, embed_size, filter_num, filter_size, dropout_rate)
        self.attention_rnn = a.TextRNNWithAttention(cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate)
        
        self.g_e = e.GroupEnhance(self.hyper_training, self.hyper_model, g)
        self.dropout = nn.Dropout(dropout_rate)
        self.k_pool = nn.AdaptiveMaxPool1d(k)  # k-max-pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # max_pooling

        # 根据新的特征维度调整卷积层
        self.conv1 = nn.Conv1d(in_channels=2 * 100, out_channels=20, kernel_size=3, padding='same')  # f 为每个特征图的通道数

        self.output_layer = nn.Linear(20 * k, class_num)  # 卷积后
        self.softmax = nn.Softmax(dim=1)
        self.g = g

    def forward(self, x):
        """
        :param x: 文本
        :return: 二分类结果
        """
        # 替换原 CNN 和 RNN 为带有注意力机制的版本
        attention_cnn_output = self.attention_cnn(x)  # (B, f, l)
        attention_rnn_output, _ = self.attention_rnn(x)  # (B, f, l)
        
        # 特征增强
        cge = self.g_e(attention_cnn_output)  # 增强后的局部特征: (B, f, l)
        rge = self.g_e(attention_rnn_output)  # 增强后的全局特征: (B, f, l)
        
        # 串联增强后的特征
        _cat = torch.cat((self.k_pool(cge), self.k_pool(rge)), dim=1)  # (B, 2*f, k)
        
        # 卷积
        _ = F.relu(self.conv1(self.dropout(_cat)))  # (B, out_channels, k)
        _ = _.view(_.shape[0], -1)  # (B, out_channels * k) # 展平
        _ = self.output_layer(_)  # (B, class_num)
        _ = self.softmax(_)  # (B, class_num)
        
        return _
