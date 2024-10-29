import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import TextCNN as c
import TextRNN as r
import TextGE as e

class ge(nn.Module):
    def __init__(self, cell, vocab_size, embed_size, filter_num, filter_size, hidden_dim, num_layers, class_num,
                 dropout_rate, g, k, hyper_training=False, hyper_model=None):
        super(ge, self).__init__()
        self.hyper_model = hyper_model
        self.hyper_training = hyper_training
        self.cnn = c.TextCNN(vocab_size, embed_size, filter_num, filter_size, dropout_rate)
        self.rnn = r.TextRNN(cell, vocab_size, embed_size, hidden_dim, num_layers, dropout_rate)
        self.g_e = e.GroupEnhance(self.hyper_training, self.hyper_model, g)
        self.dropout = nn.Dropout(dropout_rate)
        self.k_pool = nn.AdaptiveMaxPool1d(k) # k-max-pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1) # max_pooling

        self.conv1 = nn.Conv1d(in_channels=4 * hidden_dim, out_channels=20, kernel_size=3, padding='same')

        self.output_layer = nn.Linear(filter_num * 4, class_num)  # 不含GE模块
        self.output_layer1 = nn.Linear(20 * k, class_num)  # 卷积后
        self.softmax = nn.Softmax(dim=1)
        self.g = g

    def forward(self, x):
        """
        :param x:文本
        :return: 二分类结果
        """
        cnn_res = self.cnn(x)  # (B,filter_num * length(filter_size),L)
        rnn_res = self.rnn(x)  # (B, H*2, L） H：hidden_dim，NL：num_layers
        # 特征增强
        cge = self.g_e(cnn_res)  # 增强后的局部特征:(B,f,L),f=filter_num * length(filter_size)=hidden_dim*2
        rge = self.g_e(rnn_res)  # 增强后的全局特征:(B,f,L)
        """不增强/K增强，卷积后，输入output_layer1"""
        # _cat = torch.cat((cnn_res, rnn_res), dim=1) # (B, H*4, L） # 不增强+串联
        _cat = torch.cat((self.k_pool(cge), self.k_pool(rge)), dim=1) # K增强+串联(B, H*4, k）,k=10
        _ = F.relu(self.k_pool(self.dropout(self.conv1(_cat))))  # 串联卷积(B,out_channels,k),out_channels=20,k=10
        _ = _.view(_.shape[0], -1)  # (B,out_channels*k) # 展平
        _ = self.output_layer1(_)
        _ = self.softmax(_)
        return _

    def hyper_train(self):
        self.g_e = e.GroupEnhance(True, self.hyper_model, self.g)

# if __name__ == "__main__":
#     x = torch.rand((2, 10))
#     ge = ge("bi-gru",20,20,4,[3],2,1,2,0.5,1,1)
#     res = ge(x)
#     # print(x)
#     print(res)
