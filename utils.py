# -*-coding:utf-8 -*-
"""
@project:GCN
@author:Kun_J
@file:utils.py
@ide:Pycharm
@time:2020-06-18 22:37:12
@month:六月
"""
from sklearn.preprocessing import normalize
import numpy as np
import os
def evaluate_metrics(confusion_matrix):
    # https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    # Sensitivity, hit rate, recall, or
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
    # true positive rate，Sensitivity, hit rate, recall
    TPR = TP / (TP + FN)  # 召回率
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP) # 精确率
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    # ACC_micro = (sum(TP) + sum(TN)) / (sum(TP) + sum(FP) + sum(FN) + sum(TN))
    ACC_macro = np.mean(ACC)  # to get a sense of effectiveness of our method on the small classes we computed this average (macro-average)

    return ACC_macro, ACC, TPR, TNR, PPV

def batch_data(X, y, batch_size):
    shuffle = np.random.permutation(len(X))
    start = 0
    X = X[shuffle]
    y = y[shuffle]
    while start + batch_size <= len(X):
        yield X[start: start + batch_size], y[start: start + batch_size]
        start += batch_size

def mkdir(dir_path):
    # 存在此目录则删除，不存在则创建
    if os.path.exists(dir_path):
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                for root2, _, files in os.walk(os.path.join(root, dir)):
                    for f in files:
                        os.remove(os.path.join(root2, f))
        print("{} -> 删除目录、子目录、子文件成功!".format(dir_path))

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("{} -> 目录创建成功！")

def best_result(TXT_name): # 用于选取所有epoch的最优结果，主要关注S类的SEN和PPV，以及V类的PPV
    result = np.loadtxt(TXT_name)
    # print(result, type(result), result.shape)
    for res in result:
        if res[4] > 0.93 and res[6] > 0.94 and res[10] >0.99:
            print(res)

# mkdir('G:/ECG_data/Abalation2/model')

def Standard(X):
    x_normalize = normalize(X, norm='l2', axis=1) # 对每个样本进行标准化
    return x_normalize