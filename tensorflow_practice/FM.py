# -*- coding: utf-8 -*-
# @Time       : 2019/4/28 21:12
# @Author     : chensi
# @File       : FM.py
# @Software   : PyCharm
# @Desciption : None

import tensorflow as tf


class FM():
    def __init__(self, params):
        self.params = params

    def build_model(self, shape):
        self.x = tf.placeholder('float', shape=shape)
        self.y = tf.placeholder('float', shape=shape)
        self.w0 = tf.Variable(tf.zeros([1]))
        self.w = tf.Variable(tf.zeros([1]))
        self.v = tf.Variable(initial_value=tf.random_normal(shape=shape, mean=0, stddev=0.01))
        self.linear_terms = tf.add(tf.reduce_sum(tf.multiply(self.w, self.x), 1, keep_dims=True), self.w0)
        self.pair_interactions = 0.5 * tf.reduce_sum(tf.subtract(tf.pow(tf.matmul(self.x, tf.transpose(self.v)), 2),
                                                                 tf.matmul(tf.pow(self.x, 2),
                                                                           tf.transpose(tf.pow(self.v, 2)))), axis=1,
                                                     keep_dims=True)
        self.y_hat = self.linear_terms + self.pair_interactions
