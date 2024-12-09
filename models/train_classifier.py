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

class Classifier():
	
    def __init__(self, ):
        pass


    def train(self, args, data_helper):

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
                os.makedirs('parameters', exist_ok=True)
                torch.save(model.state_dict(), 'parameters/GE-' + str(epoch) + '.pkl')
                
        logger.info('best acc: {:.4f}'.format(best_acc))
        print('best acc: {:.4f}'.format(best_acc))