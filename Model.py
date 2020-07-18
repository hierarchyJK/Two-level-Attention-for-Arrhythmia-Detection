# -*-coding:utf-8 -*-
"""
@project:GCN
@author:Kun_J
@file:s2s_ECG.py
@ide:Pycharm
@time:2020-05-24 11:19:53
@month:五月
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
from extract_data import read_mitbih
from utils import evaluate_metrics, batch_data
random.seed(654)

class model():
    def model_input(self, max_time, input_depth):
        inputs = tf.placeholder(tf.float32, [None, max_time, input_depth], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')

        dec_inputs = tf.placeholder(tf.int32, [None, None], name='output')

        return inputs, targets, dec_inputs


    def SE_net(self, se_input, r):
        shape = se_input.get_shape().as_list()
        global_avg_pooling1D = tf.layers.average_pooling1d(inputs=se_input, pool_size=shape[1], strides=shape[1], padding='same')
        down = tf.layers.dense(inputs=global_avg_pooling1D, units=global_avg_pooling1D.get_shape().as_list()[2]//r, activation=tf.nn.relu)
        up = tf.layers.dense(inputs=down, units=global_avg_pooling1D.get_shape().as_list()[2], activation=tf.nn.sigmoid)

        return up
    def CNN_embedding(self, inputs, n_channels, input_depth, max_time):
        _inputs = tf.reshape(inputs, [-1, n_channels, input_depth // n_channels])  # 需要修改 shape = [None, 10, 28]

        conv_1 = tf.layers.conv1d(inputs=_inputs, filters=32, kernel_size=2, strides=1, padding='same',
                                  activation=tf.nn.relu)  # shape=[None, 10, 32]
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv_1, pool_size=2, strides=2,
                                             padding='same')  # shape = [None, 5, 32]
        up1 = self.SE_net(max_pool_1, 4)
        max_pool_1 = max_pool_1 * up1

        conv_2 = tf.layers.conv1d(inputs=max_pool_1, filters=64, kernel_size=2, strides=1, padding='same',
                                  activation=tf.nn.relu)  # shape = [None, 5, 64]
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv_2, pool_size=2, strides=2, padding='same')  # shape = [3, 64]
        up2 = self.SE_net(max_pool_2, 8)
        max_pool_2 = max_pool_2 * up2

        conv_3 = tf.layers.conv1d(inputs=max_pool_2, filters=128, kernel_size=2, strides=1, padding='same',
                                  activation=tf.nn.relu)  # shape = [None, 3, 128]

        # shape = conv_3.get_shape().as_list()
        up3 = self.SE_net(conv_3, 16)
        SE_conv3 = conv_3 * up3
        shape = SE_conv3.get_shape().as_list()

        data_input_embed = tf.reshape(SE_conv3, (-1, max_time, shape[1] * shape[2]))

        return data_input_embed


    def encoding_layer(self, num_units, max_time, batch_size, data_input_embed, bidirectional):
        with tf.variable_scope("encoding") as encoding_scope:
            if not bidirectional:
                lstm_enc = rnn.LSTMCell(num_units)
                encoder_output, last_state = tf.nn.dynamic_rnn(lstm_enc, inputs=data_input_embed, dtype=tf.float32)
            else:
                # Using a bidirectionLSTM architecture instead
                enc_fw_cell = rnn.LSTMCell(num_units)
                enc_bw_cell = rnn.LSTMCell(num_units)

                ((enc_fw_out, enc_bw_out), (enc_fw_final_state, enc_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=enc_fw_cell,
                    cell_bw=enc_bw_cell,
                    inputs=data_input_embed,
                    sequence_length=[max_time] * batch_size,
                    dtype=tf.float32
                )
                encoder_output = tf.concat((enc_fw_out, enc_bw_out), 2)

                enc_fin_c = tf.concat((enc_fw_final_state.c, enc_bw_final_state.c), 1)
                enc_fin_h = tf.concat((enc_fw_final_state.h, enc_bw_final_state.h), 1)
                last_state = rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)

        return encoder_output, last_state

    def decoding_layer_train(self, num_units, max_time, batch_size, char2numY, data_output_embed, encoder_output, last_state, bidirectional):
        if not bidirectional:
            decoder_cell = rnn.LSTMCell(num_units)
        else:
            decoder_cell = rnn.LSTMCell(2 * num_units)
        training_helper = seq2seq.TrainingHelper(inputs=data_output_embed,
                                                 sequence_length=[max_time] * batch_size,
                                                 time_major=False)

        attention_mechanism = seq2seq.BahdanauAttention(num_units=num_units,
                                                        memory=encoder_output,
                                                        memory_sequence_length=[max_time] * batch_size)

        attention_cell = seq2seq.AttentionWrapper(cell=decoder_cell,
                                                  attention_mechanism=attention_mechanism,
                                                  attention_layer_size=num_units)

        decoder_initial_state = attention_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=last_state)
        output_layer = tf.layers.Dense(len(char2numY) - 2,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        training_decoder = seq2seq.BasicDecoder(cell=attention_cell,
                                                helper=training_helper,
                                                initial_state=decoder_initial_state, output_layer=output_layer)

        train_outputs, _, _ = seq2seq.dynamic_decode(decoder=training_decoder,
                                                     impute_finished=True,
                                                     maximum_iterations=max_time)

        return train_outputs

    def decoding_layer_inference(self, num_units, max_time, batch_size, char2numY, output_embedding, encoder_output, last_state, bidirectional):
        if not bidirectional:
            decoder_cell = rnn.LSTMCell(num_units)
        else:
            decoder_cell = rnn.LSTMCell(2 * num_units)
        infer_helper = seq2seq.GreedyEmbeddingHelper(output_embedding, # Notice that different between data_output_embed
                                                     tf.fill([batch_size], char2numY['<GO>']),
                                                     char2numY['<EOS>']
                                                     )
        attention_mechanism = seq2seq.BahdanauAttention(num_units=num_units, memory=encoder_output,
                                                        memory_sequence_length=[max_time] * batch_size)
        attention_cell = seq2seq.AttentionWrapper(cell=decoder_cell,
                                                  attention_mechanism=attention_mechanism,
                                                  attention_layer_size=num_units,
                                                  )
        state = attention_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        state = state.clone(cell_state=last_state)
        output_layer = tf.layers.Dense(len(char2numY) - 2,
                                       kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        decoder = seq2seq.BasicDecoder(cell=attention_cell,
                                       helper=infer_helper,
                                       initial_state=state,
                                       output_layer=output_layer)

        infer_outputs, _, _ = seq2seq.dynamic_decode(decoder=decoder,
                                                     impute_finished=True,
                                                     maximum_iterations=max_time)

        return infer_outputs

    def decoding_layer(self, num_units, max_time, batch_size, char2numY, encoder_output, last_state, dec_inputs, bidirectional):
        embed_size = 10
        # decoderEmbedding layers
        output_embedding = tf.Variable(tf.random_uniform((len(char2numY), embed_size), -1.0, 1.0), name='dec_embedding')
        data_output_embed = tf.nn.embedding_lookup(output_embedding, dec_inputs)
        with tf.variable_scope("decode"):
            train_outputs = self.decoding_layer_train(num_units=num_units,
                                                      max_time=max_time,
                                                      batch_size=batch_size,
                                                      char2numY=char2numY,
                                                      data_output_embed=data_output_embed,
                                                      encoder_output=encoder_output,
                                                      last_state=last_state,
                                                      bidirectional=bidirectional)
        with tf.variable_scope("decode", reuse=True):
            infer_outputs = self.decoding_layer_inference(num_units=num_units,
                                                          max_time=max_time,
                                                          batch_size=batch_size,
                                                          char2numY=char2numY,
                                                          output_embedding=output_embedding,
                                                          encoder_output=encoder_output,
                                                          last_state=last_state,
                                                          bidirectional=bidirectional)
        train_logits = tf.identity(train_outputs.rnn_output)
        infer_logits = tf.identity(infer_outputs.rnn_output)
        return train_logits, infer_logits

    def Attention_seq2seq(self, num_units, batch_size, char2numY, inputs, dec_inputs, n_channels, input_depth, max_time, bidirectional):

        data_input_embed = self.CNN_embedding(inputs=inputs,
                                              n_channels=n_channels,
                                              input_depth=input_depth,
                                              max_time=max_time)

        encoder_output, last_state = self.encoding_layer(num_units=num_units,
                                                         max_time=max_time,
                                                         batch_size=batch_size,
                                                         data_input_embed=data_input_embed,
                                                         bidirectional=bidirectional)

        train_logits, infer_logits = self.decoding_layer(num_units=num_units,
                                                         max_time=max_time,
                                                         batch_size=batch_size,
                                                         char2numY=char2numY,
                                                         encoder_output=encoder_output,
                                                         last_state=last_state,
                                                         dec_inputs=dec_inputs,
                                                         bidirectional=bidirectional)

        return train_logits, infer_logits

