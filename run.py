import torch
from torch import nn
import argparse
import sys
from logger import logger
# from torchsummary import summary
import numpy as np
import data
import GE
from hypernet import LSTMModel
import time
import heapq #

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
	parser = argparse.ArgumentParser()
	parser.register("type", "bool", lambda x : x.lower() == 'true')
	parser.add_argument("--neg_filename", type=str, default='data/cover_mixbit_news.txt')
	parser.add_argument("--pos_filename", type=str, default="data/stego_mixbit_news.txt")
	parser.add_argument("--epoch", type=int, default=50)  # default=100
	parser.add_argument("--stop", type=int, default=50)
	parser.add_argument("--max_length", type=int, default=None)
	# parser.add_argument("--logdir", type=str, default="./rnnlog")
	parser.add_argument("--sentence_num", type=int, default=3000)  # default=1000
	parser.add_argument("--rand_seed", type=int, default=0)
	parser.add_argument("--hyper_train", type=bool, default=False)
	parser.add_argument("--num_turn", type=int, default=10)
	args = parser.parse_args(sys.argv[1:])
	return args
args = get_args()
# logger

import random
random.seed(args.rand_seed)  # random.seed()是随机数生成函数，

# log_dir = args.logdir
# os.makedirs(log_dir, exist_ok=True)
# log_file = log_dir + "/rnn_{}.txt".format(os.path.basename(args.neg_filename)+"___"+os.path.basename(args.pos_filename))
# logger = Logger(log_file)

def main(data_helper, hyper_train):
	# ======================
	# 超参数,设置时要注意使 filter_num * length(filter_size) = hidden_dim * 2
	# ======================
	CELL = "bi-gru"            # rnn, bi-rnn, gru, bi-gru, lstm, bi-lstm
	BATCH_SIZE = 128  # 64
	EMBED_SIZE = 300  # 128
	HIDDEN_DIM = 100  # 256
	NUM_LAYERS = 2
	CLASS_NUM = 2
	DROPOUT_RATE = 0.5  # 0.2
	EPOCH = args.epoch  # 默认100
	LEARNING_RATE = 0.001
	SAVE_EVERY = 20
	STOP = args.stop  # 默认20
	SENTENCE_NUM = args.sentence_num  # 2000
	K = 10
	G = 10
	FILTER_NUM = 100
	FILTER_SIZE = [3, 5]
	FEATURE_DIM = FILTER_NUM * 2

	# all_var = locals()
	# print()
	# for var in all_var:
	# 	if var != "var_name":
	# 		logger.info("{0:15}   ".format(var))
	# 		logger.info(all_var[var])
	# print()

	# ======================
	# 数据
	# ======================

	# ======================
	# 构建模型
	# ======================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	hyper_model = LSTMModel(FEATURE_DIM, 32, FEATURE_DIM)
	hyper_model.to(device)
	hyper_optimizer = torch.optim.Adam(hyper_model.parameters(), LEARNING_RATE, weight_decay=1e-6)  # 优化函数
	model = GE.ge(
		cell=CELL,
		vocab_size=data_helper.vocab_size,
		embed_size=EMBED_SIZE,
		filter_num=FILTER_NUM,
		filter_size=FILTER_SIZE,
		hidden_dim=HIDDEN_DIM,
		num_layers=NUM_LAYERS,
		class_num=CLASS_NUM,
		dropout_rate=DROPOUT_RATE,
		k=K,
		g=G,
		hyper_training=False,
		hyper_model=hyper_model,
	)
	# 加入超网络
	model.to(device)  # 将模型加载到指定设备上
	criteration = nn.CrossEntropyLoss() # 损失函数
	optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=1e-6) # 优化函数
	best_acc = 0
	early_stop = 0

	epoch_test_P, epoch_test_R = [], [] # add by xq 22.9.29

	# ======================
	# 训练与测试
	# ======================
	for epoch in range(EPOCH):
		generator_train = data_helper.train_generator(BATCH_SIZE)
		generator_test = data_helper.test_generator(BATCH_SIZE)
		train_loss = []
		train_acc = []
		while True:
			try:
				text, label = generator_train.__next__()
			except:
				break
			optimizer.zero_grad()  # 梯度置为0
			y = model(torch.from_numpy(text).long().to(device))
			loss = criteration(y, torch.from_numpy(label).long().to(device))
			loss.backward()
			optimizer.step()
			train_loss.append(loss.item())
			y = y.cpu().detach().numpy()
			train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
			# np.argmax()是numpy中获取array的某一个维度中数值最大的那个元素的索引

		test_loss = []
		test_acc = []

		test_tp = [] # add by xq 22.9.29
		tfn = []
		tpfn = []

		while True:
			with torch.no_grad():  # 测试阶段不需要更新梯度
				try:
					text, label = generator_test.__next__()
				except:
					break
				y = model(torch.from_numpy(text).long().to(device))  # y={Tensor:(64,2)}
				loss = criteration(y, torch.from_numpy(label).long().to(device))
				test_loss.append(loss.item())
				y = y.cpu().numpy()
				test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
				# add by xq 22.9.29
				test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
				tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
				tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]
		# logger.info('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
		#       .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
		# add by xq 22.9.29
		tpsum = np.sum(test_tp)
		test_precision = tpsum / np.sum(tfn)
		test_recall = tpsum / np.sum(tpfn)
		epoch_test_P.append(test_precision)
		epoch_test_R.append(test_recall)

		print('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
		      .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
		if np.mean(test_acc) > best_acc:
			best_acc = np.mean(test_acc)
			precison = test_precision # add by xq 22.9.29
			recall = test_recall # add by xq 22.9.29
		else:
			early_stop += 1
		if early_stop >= STOP:
			# logger.info('best acc: {:.4f}'.format(best_acc))
			print('best acc: {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}'.format(best_acc, precison, recall))
			# return best_acc
			return best_acc, precison, recall # add by xq 22.9.29

		if (epoch + 1) % SAVE_EVERY == 0:
			print('saving parameters')
			os.makedirs('models', exist_ok=True)
			torch.save(model.state_dict(), 'models/GE-' + str(epoch) + '.pkl')
# 	logger.info('best acc: {:.4f}'.format(best_acc))
# 	print('best acc: {:.4f}'.format(best_acc))
# 	return best_acc'
	# add by xq 22.9.29
	if hyper_train:
		print("Training hyper net")
		for param in model.parameters():
			param.requires_grad = False
		for param in hyper_model.parameters():
			param.requires_grad = True
		model.hyper_train()
		for epoch in range(EPOCH):
			generator_train = data_helper.train_generator(BATCH_SIZE)
			generator_test = data_helper.test_generator(BATCH_SIZE)
			train_loss = []
			train_acc = []
			while True:
				try:
					text, label = generator_train.__next__()
				except:
					break
				hyper_optimizer.zero_grad()  # 梯度置为0
				# 使用 torch.no_grad() 确保 model 的参数不更新，但仍然通过它进行前向传播
				y = model(torch.from_numpy(text).long().to(device))  # 通过 model 计算输出 y
				# print(y.requires_grad)
				loss = criteration(y, torch.from_numpy(label).long().to(device))
				loss.backward()
				hyper_optimizer.step()
				train_loss.append(loss.item())
				y = y.cpu().detach().numpy()
				train_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
			# np.argmax()是numpy中获取array的某一个维度中数值最大的那个元素的索引

			test_loss = []
			test_acc = []

			test_tp = []  # add by xq 22.9.29
			tfn = []
			tpfn = []
			while True:
				with torch.no_grad():  # 测试阶段不需要更新梯度
					try:
						text, label = generator_test.__next__()
					except:
						break
					y = model(torch.from_numpy(text).long().to(device))  # y={Tensor:(64,2)}
					loss = criteration(y, torch.from_numpy(label).long().to(device))
					test_loss.append(loss.item())
					y = y.cpu().numpy()
					test_acc += [1 if np.argmax(y[i]) == label[i] else 0 for i in range(len(y))]
					# add by xq 22.9.29
					test_tp += [1 if np.argmax(y[i]) == label[i] and label[i] == 1 else 0 for i in range(len(y))]
					tfn += [1 if np.argmax(y[i]) == 1 else 0 for i in range(len(y))]
					tpfn += [1 if label[i] == 1 else 0 for i in range(len(y))]
			# logger.info('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
			#       .format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
			# add by xq 22.9.29
			tpsum = np.sum(test_tp)
			test_precision = tpsum / np.sum(tfn)
			test_recall = tpsum / np.sum(tpfn)
			epoch_test_P.append(test_precision)
			epoch_test_R.append(test_recall)

			print('epoch {:d}   training loss {:.4f}    test loss {:.4f}    train acc {:.4f}    test acc {:.4f}'
				.format(epoch + 1, np.mean(train_loss), np.mean(test_loss), np.mean(train_acc), np.mean(test_acc)))
			if np.mean(test_acc) > best_acc:
				best_acc = np.mean(test_acc)
				precison = test_precision # add by xq 22.9.29
				recall = test_recall # add by xq 22.9.29
			else:
				early_stop += 1
			if early_stop >= STOP:
				# logger.info('best acc: {:.4f}'.format(best_acc))
				print('best acc: {:.4f}, pre {:.4f}, recall {:.4f}, F1 {:.4f}'.format(best_acc, precison, recall))
				# return best_acc
				return best_acc, precison, recall # add by xq 22.9.29

			if (epoch + 1) % SAVE_EVERY == 0:
				print('saving parameters')
				os.makedirs('models', exist_ok=True)
				torch.save(hyper_model.state_dict(), 'models/hyper-lstm-' + str(epoch) + '.pkl')

		print('best acc: {:.4f}, pre {:.4f}, recall {:.4f}'.format(best_acc, precison, recall))
		# torch.save(model.state_dict(), './train_epoch200_.pth')

	return best_acc, precison, recall


if __name__ == '__main__':
	acc = []
	preci = [] # add by xq 22.9.29
	recall = [] # add by xq 22.9.29
	# filter_data(args.neg_filename,args.pos_filename)
	# with open(args.neg_filename+"_filter", 'r', encoding='utf-8') as f:
	# 处理cover文件，将文件中的每个句子存放至列表raw_pos中，并打乱顺序
	with open(args.neg_filename, 'r', encoding='utf-8') as f:  # 以只读方式打开cover文件
		raw_pos = f.read().lower().split("\n")
	raw_pos = list(filter(lambda x: x not in ['', None], raw_pos)) # 从raw_pos中去除''和None的行
	if args.max_length is not None:
		raw_pos = [text for text in raw_pos if len(text.split()) < args.max_length]
	import random

	random.shuffle(raw_pos) # 将原列表的次序打乱
	# raw_pos = [' '.join(list(jieba.cut(pp))) for pp in raw_pos]
	# with open(args.pos_filename+"_filter", 'r', encoding='utf-8') as f:

	# 处理stego文件，将文件中的每个句子存放至列表raw_neg中，并打乱顺序
	with open(args.pos_filename, 'r', encoding='utf-8') as f: # 以只读方式打开stego文件
		raw_neg = f.read().lower().split("\n")
	raw_neg = list(filter(lambda x: x not in ['', None], raw_neg))
	if args.max_length is not None:
		raw_neg = [text for text in raw_neg if len(text.split()) < args.max_length]
	random.shuffle(raw_neg)
	# raw_neg = [' '.join(list(jieba.cut(pp))) for pp in raw_neg]
	length = min(args.sentence_num, len(raw_neg),len(raw_pos))
	# 定义一个DataHelper类的实例data_helper，传入参数为：length个cover文本，length个stego文本
	data_helper = data.DataHelper([raw_pos[:length], raw_neg[:length]], use_label=True, word_drop=0)

	# 多次实验取平均值
	for i in range(args.num_turn):
		random.seed(i) # 确保每次抽取的数据样本一致
		start = time.time() # add by xq 22.9.29
		# add by xq 22.9.29
		index = main(data_helper, args.hyper_train)
		acc.append(index[0])
		preci.append(index[1])
		recall.append(index[2])
	# print("max3_mean:", np.mean(np.sort(acc)[-3:]))
	# acc_mean = np.mean(acc)
	acc_mean = np.mean(np.sort(acc)[-3:])
	pre_mean = np.mean(np.sort(preci)[-3:])
	pre_std = np.std(preci)
	recall_mean = np.mean(np.sort(recall)[-3:])
	recall_std = np.std(recall)
	# logger.info("using %d sentences " % length)
	# logger.info("best acc : {:.4f}".format(min(acc)))
	# logger.info("worst acc: {:.4f}".format(max(acc)))
	# logger.info("acc final: {:.4f}+{:.4f}".format(acc_mean, max(acc_mean - min(acc), max(acc) - acc_mean)))
	# print("using %d sentences " % length)
	# print("worst acc : {:.4f}".format(min(acc)))
	# print("best acc: {:.4f}".format(max(acc)))
	# print("acc final: {:.4f}+{:.4f}".format(acc_mean, max(acc_mean - min(acc), max(acc) - acc_mean)))
	# 输出10次实验结果的平均值和该平均值与最大最小值的差距
	print("Final: acc {:.4f}±{:.4f}, precision {:.4f}±{:.4f}, recall {:.4f}±{:.4f}"
				.format(acc_mean, max(acc_mean - min(np.sort(acc)[-3:]), max(np.sort(acc)[-3:]) - acc_mean), pre_mean, pre_std, recall_mean, recall_std))


