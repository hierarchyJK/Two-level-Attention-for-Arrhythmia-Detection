# -*-coding:utf-8 -*-
"""
@project:GCN
@author:Kun_J
@file:utils.py
@ide:Pycharm
@time:2020-06-18 22:37:12
@month:六月
"""
import numpy as np

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