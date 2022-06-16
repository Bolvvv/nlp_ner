# -*- coding: utf-8 -*-
import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from gensim.models import KeyedVectors
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

#设置cuda使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#数据路径
train_data_path = './CONLL2003/train.txt'
valid_data_path = './CONLL2003/valid.txt'
test_data_path = './CONLL2003/test.txt'
WORD2VEC_MATRIX_PATH = './glove.42B.300d.txt'#需提前准备

CHECK_POINT_PATH = './model.pkl'#需提前准备

#********************词向量载入，第一次启动项目时使用方法1生成二进制词向量，后面使用方法2载入词向量（方法2可以加快载入速度）********************
#方法1:
glove_model = KeyedVectors.load_word2vec_format(WORD2VEC_MATRIX_PATH, no_header=True)
glove_model.init_sims(replace=True)
glove_model.save(WORD2VEC_MATRIX_PATH.replace(".txt", ".bin"))

#方法2:
# glove_model = KeyedVectors.load(WORD2VEC_MATRIX_PATH.replace(".txt", ".bin"), mmap='r')
#*********************************************************************************************************

#超参数设定
window_size = 2
N, D_in, H, D_out = 1, 900, 300, 5
LR = 0.01

def classifier(word_class):
    y_hat = [0, 0, 0, 0, 0]
    #注意这里是字幕O不是数字0
    if word_class == 'O':
        y_hat = [1, 0, 0, 0, 0]
    elif word_class == 'B-PER' or word_class == 'I-PER':
        y_hat = [0, 1, 0, 0, 0]
    elif word_class == 'B-LOC' or word_class == 'I-LOC':
        y_hat = [0, 0, 1, 0, 0]
    elif word_class == 'B-ORG' or word_class == 'I-ORG':
        y_hat = [0, 0, 0, 1, 0]
    elif word_class == 'B-MISC' or word_class == 'I-MISC':
        y_hat = [0, 0, 0, 0, 1]
    else:
        print("原始文本类别错误")
    return y_hat

def classifier_trans(numpy_array_class):
    o = 0
    per = 1
    loc = 2
    org = 3
    misc = 4

    if numpy_array_class == o:
        return "O"
    elif numpy_array_class == per:
        return "PER"
    elif numpy_array_class == loc:
        return "LOC"
    elif numpy_array_class == org:
        return "ORG"
    elif numpy_array_class == misc:
        return "MISC"
    else:
        print("原始文本类别错误")
        return "WRONG CLASS"
        


def get_data(index, window_size, word_list, word_class_list):
    x = np.array([])#输入向量，维度为900,即窗口为3，一个词向量对应维度为300
    for i in range(index-window_size, index + window_size+1):
        if i < 0 or i >= len(word_list):
            continue
        else:
            try:
                tmp = glove_model[word_list[i]]
                x = np.concatenate((x, tmp))
            except KeyError:
                #出现此错误表示在词向量矩阵中未找到对应词的向量，将此词的向量设置为0
                tmp = np.zeros(300)
                x = np.concatenate((x, tmp))
    y = classifier(word_class_list[index])
    #进行格式转换
    x = torch.from_numpy(x).float()
    y = torch.tensor(y).float()
    return x,y

def preprocessing(data_path):
    """
    由于仅需要数据集中的单词和类别，因此只存储单词和其对应的类别
    类别包括 PER, LOC, ORG, MISC, 0
    其中除了'0'类，其他类别都有前缀'B-'和'I-，在分类时暂不考虑差别
    此命名方法为BOI命名法
    """
    f = open(data_path, 'r')
    word_list = []
    word_class_list = []
    for line in f.readlines() :
        tmp_list = line.split(' ')
        if line == '\n':
            continue
        word_list.append(tmp_list[0])
        word_class_list.append(tmp_list[3].strip('\n'))
    f.close()
    return word_list, word_class_list

class NERModel(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NERModel, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        m = torch.nn.Hardtanh(min_val=-1, max_val=1, inplace=False)
        h_hardtanh = m(self.linear1(x))
        n = torch.nn.Softmax(dim=0)
        y_pred = n(self.linear2(h_hardtanh))
        return y_pred

def train(data_path, model, device, Lr):
    optimizer = torch.optim.SGD(model.parameters(), lr=Lr)
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    model.to(device)
    word_list, word_class_list = preprocessing(train_data_path)#获取预处理词汇
    startTime=time.time()
    for i in range(1,len(word_list)-1):
        for t in range(1):
            model.train()
            x, y = get_data(i, 1, word_list, word_class_list)
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if i % 100 == 99:
                print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        checkpoint = {
            'state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'epoch': i
        }
    #保存网络
    torch.save(checkpoint, CHECK_POINT_PATH) 
    print("training took %f seconds" % (time.time() - startTime))

def valid(data_path, model, checkpoint_path, device, valid_method):
    word_list, word_class_list = preprocessing(train_data_path)#获取预处理词汇
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    #载入模型
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    model.eval()#设置为评价模式
    right_count = 0 #预测正确的数量
    all_count = 0 #单词总量
    startTime=time.time()#计时
    for i in range(1,len(word_list)-1):
        #当评价方式设置为1时，会去除类别为'O'的单词，以免'O'类单词数量过多造成准确率虚高的问题
        if valid_method == 1:
            if word_class_list[i] == 'O':
                continue

        x, y = get_data(i, 1, word_list, word_class_list)
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        y_trans = y.cpu().detach().numpy()
        y_trans_pred = y_pred.cpu().detach().numpy()

        print("预测结果:"+classifier_trans(np.argmax(y_trans_pred))+ " "+"实际结果:"+classifier_trans(np.argmax(y_trans))+" "+str(np.argmax(y_trans) == np.argmax(y_trans_pred)))
        all_count += 1 #单词总量加1
        #预测正确，数量加1
        if np.argmax(y_trans) == np.argmax(y_trans_pred):
            right_count+=1
        if all_count % 100 == 99:
            print("当前准确率为:%f" % (right_count/all_count))

    print("valid took %f seconds" % (time.time() - startTime))
    print("模型预测正确率：%f" % (right_count/all_count))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default='train', help="is train or valid")
    parser.add_argument("--method", type=int, default=0, help="valid method")
    opt = parser.parse_args()
    model = NERModel(D_in, H, D_out).to(device)#生成模型
    if opt.stage == 'train':
        train(train_data_path, model, device, LR)
    elif opt.stage == 'valid':
        valid(valid_data_path, model, CHECK_POINT_PATH, device, opt.method)

main()