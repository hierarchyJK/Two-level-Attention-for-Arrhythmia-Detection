# -*-coding:utf-8 -*-
"""
@project:GCN
@author:Kun_J
@file:Train.py.py
@ide:Pycharm
@time:2020-06-20 16:21:25
@month:六月
"""
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from datetime import datetime
import tensorflow as tf
import scipy.io as spio
import numpy as np
import argparse
import random
import tqdm
import time
import gc
import os
from extract_data import data_process
from utils import evaluate_metrics, batch_data, mkdir
from Model import model

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
    parser.add_argument('--checkpoint_dir', type=str, default='G:/ECG_data/model_save/model')
    parser.add_argument('--result_dir', type=str, default='G:/ECG_data/model_save/result_0')
    parser.add_argument('--ckpt_name', type=str, default='seq2seq_mitbih_DS1DS2.ckpt')
    parser.add_argument('--classes', nargs="+", type=chr, default=['N', 'S', 'V'])
    args = parser.parse_args()

    train(args)


def train(args):
    max_time = args.max_time # defaule 9
    epochs = args.epochs # 1000
    batch_size = args.batch_size # 20
    num_units = args.num_units # 128
    bidirectional = args.bidirectional
    n_oversampling = args.n_oversample
    checkpoint_dir = args.checkpoint_dir
    ckpt_name = args.ckpt_name
    test_steps = args.test_steps
    classes = args.classes  # ['N', 'S','V']
    filename = args.data_dir
    result_dir = args.result_dir # 用于保存每次结果

    X_train, y_train, X_test, y_test, n_classes, char2numY, input_depth, y_seq_length\
    = data_process(max_time, n_oversampling, classes, filename)

    my_model = model()
    inputs, targets, dec_inputs = my_model.model_input(max_time=max_time,
                                                    input_depth=input_depth)

    logits, infer_logits = my_model.Attention_seq2seq(num_units=num_units,
                                                   batch_size=batch_size,
                                                   char2numY=char2numY,
                                                   inputs=inputs,
                                                   dec_inputs=dec_inputs,
                                                   n_channels=10,
                                                   input_depth=input_depth,
                                                   max_time=max_time,
                                                   bidirectional=bidirectional)
    with tf.name_scope('optimization'):
        vars = tf.trainable_variables()
        beta = 0.001
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) * beta

        loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))

        # optimizer
        loss = tf.reduce_mean(loss + lossL2)
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)


    results = []
    def test_model():
        print('测试')
        acc_track = []
        sum_test_conf = []
        for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_test, y_test, batch_size)):
            # dec_input = np.zeros((len(source_batch), 1)) + char2numY['<GO>']

            batch_logits = sess.run(infer_logits,
                                    feed_dict={
                                        inputs: source_batch,
                                        })
            y_pred = batch_logits.argmax(axis=-1)

            y_true = target_batch[:, 1:-1]
            acc_track.append(y_pred == y_true)
            # print('测试集预测值：', y_pred)
            # print('测试集真实值：', y_true)
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()
            sum_test_conf.append(confusion_matrix(y_true, y_pred, labels=range(len(char2numY)-2)))

        sum_test_conf = np.mean(np.array(sum_test_conf, dtype=np.float32), axis=0)

        acc_avg, acc, sensitivity, specificity, PPV = evaluate_metrics(sum_test_conf)
        print("Average Accuracy is: {:>6.4f} on test set\n".format(acc_avg))

        info = ''
        for index_ in range(n_classes):
            print('\t {} rhythm -> Sensitivity(recall): {:1.4f}, Specificity: {:1.4f}, Precision(PPV): {:1.4f}, Accuracy: {:1.4f}'.format(
                classes[index_],
                sensitivity[index_],
                specificity[index_],
                PPV[index_],
                acc[index_]
            ))
            strings = "{:1.4f} {:1.4f} {:1.4f} {:1.4f}".format(sensitivity[index_], specificity[index_], PPV[index_], acc[index_])
            info += strings

        print('\n Average -> Sensitivity: {:1.4f}, Specificity: {:1.4f}, Precision: {:1.4f}, Accuracy: {:1.4f}'.format(
            np.mean(sensitivity),
            np.mean(specificity),
            np.mean(PPV),
            np.mean(acc)
        ))

        results.append(info)

        return acc_avg, acc, sensitivity, specificity, PPV



    def count_pramaters():
        print('# of params:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    count_pramaters()

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
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            test_model()
        else:
            losses = []
            loss_track = []
            for epoch_i in range(epochs):
                start_time = time.time()
                train_acc = []
                for batch_i, (source_batch, target_batch) in enumerate(batch_data(X_train, y_train, batch_size)):
                    _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                    feed_dict = {
                        inputs: source_batch,
                        dec_inputs: target_batch[:, :-2],
                        targets: target_batch[:, 1:-1]
                    })
                    loss_track.append(batch_loss)
                    train_acc.append(batch_logits.argmax(axis=-1) == target_batch[:, 1:-1])
                    # print('训练预测', batch_logits.argmax(axis=-1))
                    # print('训练真实值', target_batch[:, 1:-1])

                accuracy = np.mean(train_acc)
                print('Epoch {:3} Loss:{:>6.3f} Accuracy:{:>6.4f} Epoch duration:{:>6.3f}'.format(
                    epoch_i,
                    sum(loss_track)/len(loss_track),
                    accuracy,
                    time.time() - start_time
                ))
                losses.append(sum(loss_track) / len(loss_track))
                if (epoch_i + 1) % test_steps == 0:
                    acc_avg, acc, sensitivity, specificity, PPV = test_model() # 输出测试结果
                    print("loss:{:.4f} after {} epochs (batch_size={})".format(loss_track[-1], epoch_i + 1, batch_size))

                    save_path = os.path.join(checkpoint_dir, ckpt_name)
                    saver.save(sess, save_path)
                    print("Model saved in path:%s" % save_path)

            with open(os.path.join(result_dir, "loss.txt"), mode="w") as f: # 保存loss
                for l in losses:
                    f.write(str(l) + "\n")

            with open(os.path.join(result_dir, "infos.txt"), mode='w') as f: # 保存每一次测结果
                for info in results:
                    f.write(info + "\n")
        print(str(datetime.now()))

if __name__ == "__main__":
    time_start = time.time()
    print("=============TRAIN_STATR=============")
    main()
    print("=============TRAIN_end=============")
    time_end = time.time()
    print("跑一次耗时：{:.4f}".format(time_end - time_start))