import torch
from torch import nn


class TextCNN(nn.Module):
	def __init__(self, vocab_size, embed_size, filter_num, filter_size, dropout_rate):
		super(TextCNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embed_size) # vocab_size：当前要训练的文本中不重复单词的个数
		self.cnn_list = nn.ModuleList()
		for size in filter_size:
			self.cnn_list.append(nn.Conv1d(embed_size, filter_num, size, padding='same'))
		self.relu = nn.ReLU()
		self.max_pool = nn.AdaptiveMaxPool1d(1)
		self.dropout = nn.Dropout(dropout_rate)
		# self.output_layer = nn.Linear(filter_num * len(filter_size), class_num)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		"""
		:param x:(N,L)，N为batch_size，L为句子长度
		:return: (N,class_num) class_num是分类数，文本隐写分析最终分为stego和cover两类
		"""
		x = x.long()
		_ = self.embedding(x) # 词嵌入，（N,L,embed_size）
		_ = _.permute(0, 2, 1) # 卷积是在最后一维进行，因此要交换embed_size维和L维，在句子上进行卷积操作
		result = [] # 定义一个列表，存放每次卷积、池化后的特征值，最终列表元素个数=卷积核size的个数
		for self.cnn in self.cnn_list:
			__ = self.cnn(_) # 卷积操作
			__ = self.relu(__)
			result.append(__) # 判断第2维的维度是否为1，若是则去掉.因为池化后的第2维是1，因此这里是去掉第2维，结果是（batch_size,filter_num）

		_ = torch.cat(result, dim=1) # 将result列表中的元素在行上进行拼接
		_ = self.dropout(_)
		_ = self.softmax(_)
		return _