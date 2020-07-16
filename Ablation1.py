# -*-coding:utf-8 -*-
"""
@project:GCN
@author:Kun_J
@file:Ablation1.py.py
@ide:Pycharm
@time:2020-07-16 11:37:57
@month:七月
"""
# -*- coding:utf-8 -*-
"""
@project: ECG_Seq2Seq
@author: KunJ
@file: Abalation1.py.py
@ide: Pycharm
@time: 2020-07-16 11:36:12
@month: 七月
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn.preprocessing import MinMaxScaler
import random
import time
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import argparse
import tqdm
import gc
from utils import mkdir, batch_data, evaluate_metrics
from extract_data import read_mitbih
random.seed(654)

def data_process(max_time, n_oversampling, classes, filename):
    X_train, y_train = read_mitbih(filename, max_time, classes, max_nlabel=50000, trainset=1)
    X_test, y_test = read_mitbih(filename, max_time, classes, max_nlabel=50000, trainset=0)

    input_depth = X_train.shape[2]
    print(input_depth)  # 280
    n_channels = 10
    print("#of sequences:", len(X_train))

    classes = np.unique(y_train)  # N, S, V
    char2numY = dict(zip(classes, range(len(classes))))

    n_classes = len(classes)  # 3
    print("Class (training):", classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_train == cl)[0]))  # N:45796, S:941, V:3780

    print("Class (test):", classes)
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        print(cl, len(np.where(y_test.flatten() == cl)[0]))  # N:44196, S:1836, V:3216

    char2numY['<GO>'] = len(char2numY)  # encoder以<GO>开始输入
    # char2numY['<EOS>'] = len(char2numY) # encoder以<EOS>结尾输出
    # print('char2numY(加上<GO>):', char2numY) # {'N': 0, 'S': 1, 'V': 2, '<GO>': 3, '<EOS>':4}
    num2CharY = dict(zip(char2numY.values(), char2numY.keys()))

    y_train = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y_train]
    y_test = [[char2numY['<GO>']] + [char2numY[y_] for y_ in date] for date in y_test]
    y_test = np.asarray(y_test)
    y_train = np.array(y_train)

    x_seq_length = len(X_train[0])
    y_seq_length = len(y_train[0]) - 1

    print('-------------SMOTE----------------')
    X_train = np.reshape(X_train, [X_train.shape[0] * X_train.shape[1], -1])
    print(y_train)
    y_train = y_train[:, 1:].flatten()

    nums = []
    for cl in classes:
        ind = np.where(classes == cl)[0][0]
        nums.append(len(np.where(y_train.flatten() == ind)[0]))

    ratio = {0: nums[0], 1: n_oversampling + 1000, 2: n_oversampling}
    sm = SMOTE(random_state=2020, ratio=ratio)

    X_train, y_train = sm.fit_sample(X_train, y_train)

    X_train = X_train[:(X_train.shape[0] // max_time) * max_time, :]
    y_train = y_train[:(X_train.shape[0] // max_time) * max_time]

    X_train = np.reshape(X_train, [-1, X_test.shape[1], X_test.shape[2]])
    y_train = np.reshape(y_train, [-1, y_test.shape[1] - 1, ])

    y_train = [[char2numY['<GO>']] + [y_ for y_ in data] for data in y_train]
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


def build_network(inputs, dec_inputs, char2numY, n_channels=10, input_depth=280, num_units=128, max_time=10,
                  bidirectional=False):
    _inputs = tf.reshape(inputs, [-1, n_channels, input_depth // n_channels])
    # _inputs = tf.reshape(inputs, [-1,input_depth,n_channels])

    # #(batch*max_time, 280, 1) --> (N, 280, 18)
    conv1 = tf.layers.conv1d(inputs=_inputs, filters=32, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=128, kernel_size=2, strides=1,
                             padding='same', activation=tf.nn.relu)

    shape = conv3.get_shape().as_list()
    data_input_embed = tf.reshape(conv3, (-1, max_time, shape[1] * shape[2]))

    embed_size = 10  # 128 lstm_size # shape[1]*shape[2]

    # Embedding layers
    output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
    data_output_embed = tf.nn.embedding_lookup(output_embedding, dec_inputs)

    with tf.variable_scope("encoding") as encoding_scope:
        if not bidirectional:
            # Regular approach with LSTM units
            lstm_enc = tf.contrib.rnn.LSTMCell(num_units)
            _, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=data_input_embed, dtype=tf.float32)

        else:
            # Using a bidirectional LSTM architecture instead
            enc_fw_cell = tf.contrib.rnn.LSTMCell(num_units)
            enc_bw_cell = tf.contrib.rnn.LSTMCell(num_units)

            ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=enc_fw_cell,
                cell_bw=enc_bw_cell,
                inputs=data_input_embed,
                dtype=tf.float32)
            enc_fin_c = tf.concat((enc_fw_final.c, enc_bw_final.c), 1)
            enc_fin_h = tf.concat((enc_fw_final.h, enc_bw_final.h), 1)
            last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)

    with tf.variable_scope("decoding") as decoding_scope:
        if not bidirectional:
            lstm_dec = tf.contrib.rnn.LSTMCell(num_units)
        else:
            lstm_dec = tf.contrib.rnn.LSTMCell(2 * num_units)

        dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=data_output_embed, initial_state=last_state)

    logits = tf.layers.dense(dec_outputs, units=len(char2numY), use_bias=True)

    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--max_time', type=int, default=9)
    parser.add_argument('--test_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--data_dir', type=str, default='G:/ECG_data/s2s_mitbih_aami_DS1DS2.mat')
    parser.add_argument('--bidirectional', type=bool, default='False')
    parser.add_argument('--num_units', type=int, default=128)
    parser.add_argument('--n_oversample', type=int, default=6000)
    parser.add_argument('--checkpoint_dir', type=str, default='G:/ECG_data/Abalation1/model')
    parser.add_argument('--result_dir', type=str, default='G:/ECG_data/Abalation1/result_0')
    parser.add_argument('--ckpt_name', type=str, default='seq2seq_mitbih_DS1DS2.ckpt')
    parser.add_argument('--classes', nargs="+", type=chr, default=['N', 'S', 'V'])
    args = parser.parse_args()

    run_program(args)


def run_program(args):
    max_time = args.max_time  # defaule 9
    epochs = args.epochs  # 1000
    batch_size = args.batch_size  # 20
    num_units = args.num_units  # 128
    bidirectional = args.bidirectional
    n_oversampling = args.n_oversample
    checkpoint_dir = args.checkpoint_dir
    ckpt_name = args.ckpt_name
    test_steps = args.test_steps
    classes = args.classes  # ['N', 'S','V']
    filename = args.data_dir
    result_dir = args.result_dir  # 用于保存每次结果

    X_train, y_train, X_test, y_test, n_classes, char2numY, input_depth, y_seq_length \
        = data_process(max_time, n_oversampling, classes, filename)

    # Placeholders
    inputs = tf.placeholder(tf.float32, [None, max_time, input_depth], name='inputs')
    targets = tf.placeholder(tf.int32, (None, None), 'targets')
    dec_inputs = tf.placeholder(tf.int32, (None, None), 'output')

    logits = build_network(inputs, dec_inputs, char2numY, n_channels=10, input_depth=input_depth,
                           num_units=num_units, max_time=max_time,
                           bidirectional=bidirectional)

    with tf.name_scope("optimization"):
        # Loss function
        vars = tf.trainable_variables()
        beta = 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars
                           if 'bias' not in v.name]) * beta
        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
        # Optimizer
        loss = tf.reduce_mean(loss + lossL2)
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


    results = []
    def test_model():
        # source_batch, target_batch = next(batch_data(X_test, y_test, batch_size))
        print("测试")
        acc_track = []
        sum_test_conf = []
        for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):
            dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']
            for i in range(y_seq_length):
                batch_logits = sess.run(logits,
                                        feed_dict={inputs: source_batch, dec_inputs: dec_input})
                prediction = batch_logits[:, -1].argmax(axis=-1)
                dec_input = np.hstack([dec_input, prediction[:, None]])

            acc_track.append(dec_input[:, 1:] == target_batch[:, 1:])
            y_true = target_batch[:, 1:].flatten()
            y_pred = dec_input[:, 1:].flatten()
            sum_test_conf.append(confusion_matrix(y_true, y_pred, labels=range(len(char2numY) - 1)))

        sum_test_conf = np.mean(np.array(sum_test_conf, dtype=np.float32), axis=0)

        acc_avg, acc, sensitivity, specificity, PPV = evaluate_metrics(sum_test_conf)
        print('Average Accuracy is: {:>6.4f} on test set'.format(acc_avg))

        info = ""
        for index_ in range(n_classes):
            print("\t{} rhythm -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}".format(
                    classes[index_],
                    sensitivity[index_],
                    specificity[index_],
                    PPV[index_],
                    acc[index_]))
            strings = "{:1.4f} {:1.4f} {:1.4f} {:1.4f}".format(sensitivity[index_], specificity[index_], PPV[index_], acc[index_])
            info += strings

        results.append(info)

        print("\t Average -> Sensitivity: {:1.4f}, Specificity : {:1.4f}, Precision (PPV) : {:1.4f}, Accuracy : {:1.4f}".format(
                np.mean(sensitivity), np.mean(specificity), np.mean(PPV), np.mean(acc)))
        return acc_avg, acc, sensitivity, specificity, PPV

    def count_prameters():
        print('# of Params: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    count_prameters()

    mkdir(checkpoint_dir)
    mkdir(result_dir)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        print(str(datetime.now()))
        pre_acc_avg = 0.0
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # # Restore
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            # saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            # or 'load meta graph' and restore weights
            # saver = tf.train.import_meta_graph(ckpt_name+".meta")
            # saver.restore(session,tf.train.latest_checkpoint(checkpoint_dir))
            test_model()
        else:
            losses = []
            loss_track = []
            for epoch_i in range(epochs):
                start_time = time.time()
                train_acc = []
                for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                    _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                                           feed_dict={inputs: source_batch,
                                                                      dec_inputs: target_batch[:, :-1],
                                                                      targets: target_batch[:, 1:]})
                    loss_track.append(batch_loss)
                    train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:, 1:])

                accuracy = np.mean(train_acc)
                print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(
                    epoch_i,
                    sum(loss_track)/len(loss_track),
                    accuracy,
                    time.time() - start_time))
                losses.append(sum(loss_track) / len(loss_track))

                if epoch_i % test_steps == 0:
                    acc_avg, acc, sensitivity, specificity, PPV = test_model()

                    print('loss {:.4f} after {} epochs (batch_size={})'.format(loss_track[-1], epoch_i + 1, batch_size))
                    save_path = os.path.join(checkpoint_dir, ckpt_name)
                    saver.save(sess, save_path)
                    print("Model saved in path: %s" % save_path)

            with open(os.path.join(result_dir, "loss.txt"), mode="w") as f:  # 保存loss
                for l in losses:
                    f.write(str(l) + "\n")

            with open(os.path.join(result_dir, "infos.txt"), mode='w') as f:  # 保存每一次测结果
                for info in results:
                    f.write(info + "\n")
        print(str(datetime.now()))


if __name__ == "__main__":
    """消融实验1：CNN + Seq2Seq"""
    time_start = time.time()
    print("=============TRAIN_STATR=============")
    main()
    print("=============TRAIN_end=============")
    time_end = time.time()
    print("跑一次耗时：{:.4f}".format(time_end - time_start))

