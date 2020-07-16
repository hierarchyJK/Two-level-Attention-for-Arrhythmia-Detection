# -*-coding:utf-8 -*-
"""
@project:GCN
@author:Kun_J
@file:extract_data.py
@ide:Pycharm
@time:2020-06-18 22:26:18
@month:六月
"""
import gc
import tqdm
import numpy as np
import scipy.io as spio
from imblearn.over_sampling import SMOTE
def read_mitbih(filename, max_time, classes, max_nlabel=50000, trainset=1):
    """
    :param filename: the .mat file path
    :param max_time: the hyper-parameter of sequence length
    :param classes: ['N', 'S', 'V']
    :param max_nlabel:
    :param trainset:train or test
    :return:
    """
    samples = spio.loadmat(filename)
    print(type(samples), samples.keys())
    print("__header__: ", samples['__header__'])
    print("__version__::", samples['__version__'],'\n', "__globals__:", samples['__globals__'])
    if trainset == 1:
        samples = samples['s2s_mitbih_DS1']
    else:
        samples = samples['s2s_mitbih_DS2']

    values = samples[0]['seg_values'] # 22个人的心拍信息
    labels = samples[0]['seg_labels'] # 22个人中心拍的标签
    print(len(labels), type(labels))

    num_annotation = sum([item.shape[0] for item in values])
    print('处理前训练集样本总数：', num_annotation)

    n_seqs = num_annotation // max_time
    print("n_seqs:", n_seqs)

    print("\n-------- process all ECG-beats train data together ---------")
    data = []
    l_data = 0
    for i, item in tqdm.tqdm(enumerate(values)):
        for itm in item:
            if l_data == max_time * n_seqs:
                break
            data.append(itm[0])
            l_data += 1
    print('训练集中心拍数量：', len(data))
    print('单个心拍的长度：', data[5].shape)

    print("\n-------- process all ECG-beats labels together ---------")
    t_labels = []
    l_labels = 0
    for i, item in tqdm.tqdm(enumerate(labels)):
        for label in item[0]:
            if l_labels == n_seqs * max_time:
                break
            t_labels.append(str(label))
            l_labels += 1
    print('训练集中标签数量：', len(t_labels))

    assert len(data) == len(t_labels)

    del samples, labels, values
    gc.collect()

    data = np.asarray(data)
    shape_v = data.shape
    print('shape_v:', shape_v)
    data = np.reshape(data, [shape_v[0], -1]) #(50940, 280)
    t_labels = np.asarray(t_labels)
    print(t_labels[:10])
    _data = np.asarray([], dtype=np.float64).reshape(0, shape_v[1])
    _labels = np.asarray([], dtype=np.dtype('|S1')).reshape(0, )


    for cls in classes: # ['N', 'S', 'V']
        _label = np.where(t_labels == cls)
        permute = np.random.permutation(len(_label[0]))
        _label = _label[0][permute[:max_nlabel]]

        _data = np.concatenate((_data, data[_label]), axis=0)
        _labels = np.concatenate((_labels, t_labels[_label]))
        print(cls, len(_label), _data.shape, _labels.shape)

    data = _data[:(len(_data) // max_time) * max_time, :]
    labels = _labels[:(len(_data) // max_time) * max_time]

    print(type(data), data.shape) ## (50517, 280)

    print("-----------制作sqe2sqe的输入------------")
    data = [data[i: i + max_time] for i in range(0, len(data), max_time)]
    labels = [labels[i: i + max_time] for i in range(0, len(labels), max_time)]
    assert len(data) == len(labels)

    print("--------shuffle----------")
    permute = np.random.permutation(len(labels))
    data = np.asarray(data)
    labels = np.asarray(labels)
    data = data[permute]
    labels = labels[permute]

    print('------------seq2seq模型数据制作完毕-----------')
    print(data.shape, labels.shape, len(data))
    return data, labels

def data_process(max_time, n_oversampling, classes, filename):

    X_train, y_train = read_mitbih(filename, max_time, classes, max_nlabel=50000, trainset=1)
    X_test, y_test = read_mitbih(filename, max_time, classes, max_nlabel=50000, trainset=0)

    input_depth = X_train.shape[2]
    print(input_depth)  # 280
    n_channels = 10
    print("#of sequences:", len(X_train))

    classes = np.unique(y_train)  # N, S, V
    char2numY = dict(zip(classes, range(len(classes))))

    n_classes = len(classes) # 3
    print("Class (training):", classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_train == cl)[0])) # N:45796, S:941, V:3780

    print("Class (test):", classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_test.flatten() == cl)[0])) # N:44196, S:1836, V:3216

    char2numY['<GO>'] = len(char2numY) # encoder以<GO>开始输入
    char2numY['<EOS>'] = len(char2numY) # encoder以<EOS>结尾输出
    print('char2numY(加上<GO>):', char2numY) # {'N': 0, 'S': 1, 'V': 2, '<GO>': 3, '<EOS>':4}
    num2CharY = dict(zip(char2numY.values(), char2numY.keys()))


    y_train = [[char2numY['<GO>']] + [char2numY[y_] for y_ in data] + [char2numY['<EOS>']] for data in y_train]
    y_test = [[char2numY['<GO>']] + [char2numY[y_] for y_ in data] + [char2numY['<EOS>']] for data in y_test]

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print(y_train.shape, y_test.shape)

    x_seq_length = len(X_train[0])  # 3
    y_seq_length = len(y_train[0]) - 2  # 3

    print('-------------SMOTE----------------')
    X_train = np.reshape(X_train, [X_train.shape[0] * X_train.shape[1], -1])
    print(y_train)
    y_train = y_train[:, 1:-1].flatten()

    nums = []
    for cl in classes:
        ind = np.where(classes==cl)[0][0]
        nums.append(len(np.where(y_train.flatten() == ind)[0]))


    ratio = {0: nums[0], 1: n_oversampling + 1000, 2: n_oversampling}
    sm = SMOTE(random_state=2020, ratio=ratio)

    X_train, y_train = sm.fit_sample(X_train, y_train)

    X_train = X_train[:(X_train.shape[0] // max_time) * max_time, :]
    y_train = y_train[:(X_train.shape[0] // max_time) * max_time]

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1]-2, ])

    y_train = [[char2numY['<GO>']] + [y_ for y_ in data] + [char2numY['<EOS>']] for data in y_train]
    y_train = np.array(y_train)

    print('Classes in the training set:', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_train.flatten() == ind)[0]))

    print('------------y_train sample--------------')
    for ii in range(2):
        print(''.join([num2CharY[y_] for y_ in list(y_train[ii])]))

    print('Classes in the test set:', classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_test.flatten() == ind)[0]))

    print('------------y_test sample--------------')
    for ii in range(2):
        print(''.join([num2CharY[y_] for y_ in list(y_test[ii])]))

    return X_train, y_train, X_test, y_test, n_classes, char2numY, input_depth, y_seq_length



