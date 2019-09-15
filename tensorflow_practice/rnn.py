# -*- coding: utf-8 -*-
# @Time       : 2019/9/15 9:08
# @Author     : chensi
# @File       : rnn.py
# @Software   : PyCharm
# @Desciption : None

import numpy as np
import tensorflow as tf


def rnn_forward():
    X = [1, 2]
    state = [0, 0]
    # 隐藏层参数
    w_cell_state = np.asanyarray([[0.1, 0.2], [0.3, 0.4]])
    w_cell_input = np.asanyarray([0.5, 0.6])
    b_cell = np.asanyarray([0.1, -0.1])
    # 输出层参数
    w_output = np.asanyarray([[1.0], [2.0]])
    b_output = 0.1

    # 按时间顺序前向传播
    for t in range(len(X)):
        print("\ntime step:", t)
        print("last state:", state)
        # hidden state
        state_no_activation = np.dot(w_cell_state, state) + w_cell_input * X[t] + b_cell
        state = np.tanh(state_no_activation)
        # output
        final_output = np.dot(state, w_output) + b_output
        # print log

        print("current state:", state)
        print("final_output:", final_output)


def lstm(lstm_hidden_size, batch_size, num_steps, inputs, labels):
    # 定义lstm结构
    lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=lstm_hidden_size)
    # 生成全0的初始状态
    state = lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

    # 损失要在所有时间步上雷杰
    loss = 0
    # num_steps:最长序列长度
    for t in range(num_steps):
        #
        cur_input = inputs[t]
        # 在第一个时间步声明变量，后序时间步复用已定义的变量
        if t > 0:
            tf.get_variable_scope().reuse_variables()
            # 根据当前输入和上一时刻状态得到
            lstm_output, state = lstm(inputs=cur_input, state=state)
            final_output = tf.contrib.layers.fully_connected(lstm_output)
            # loss+=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[t],logits=final_output)


def deep_rnn(lstm_size, num_of_layers, batch_size, num_steps, inputs, labels):
    """
    构造方法：在BasicLSTMCell之上封装一层MultiRNNCell
    """
    # 定义单个循环体
    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # 在每个时间步上，将一个循环体重复num_of_layers次
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * num_of_layers)

    """
    使用带dropout的rnn：只在一个时间步的不同层使用dropout，不同时间步不用
    """
    dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(cell=lstm, output_keep_prob=0.5)  # 结点被保留的概率
    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([dropout_lstm] * num_of_layers)

    # 生成全0的初始状态
    state = stacked_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)

    # 沿时间步前向传播
    loss = 0
    for t in range(num_steps):
        cur_input = inputs[t]
        #
        if t > 0: tf.get_variable_scope().reuse_variables()
        stacked_lstm_output, state = stacked_lstm(inputs=cur_input, state=state)
        final_output = tf.contrib.layers.fully_connected(stacked_lstm_output)
        # loss+=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels[t],logits=final_output)


# def nlp():
#     from tensorflow.models.rnn.ptb import reader
