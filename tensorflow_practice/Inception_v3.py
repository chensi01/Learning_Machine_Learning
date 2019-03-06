# -*- coding: utf-8 -*-
# @File    : Inception_v3.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/6
# @Desc    :

import tensorflow as tf
import tensorflow.contrib.slim as slim

"""使用tensorflow原始API构建卷积层"""
# with tf.name_scope('conv_test'):
# 	weights = tf.get_variable(name='weights',...)
# 	biases = tf.get_variable(name='biases', ...)
# 	conv = tf.nn.conv2d()
# relu = tf.nn.relu(tf.nn.bias_add(conv,bias=biases))

"""使用Tensorflow-Slim"""
# net = slim.conv2d(input, 32, [3, 3])

"""实现部分Inception-v3的结构"""
net = "上一层的输出"
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
	with tf.variable_scope('Mixed_7c'):
		with tf.variable_scope('branch_0'):
			branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
		with tf.variable_scope('branch_1'):
			branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
			branch_1 = tf.concat(3, [slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
									 slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')])
		with tf.variable_scope('branch_2'):
			branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
			branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
			branch_2 = tf.concat(3, [slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0b_1x3'),
									 slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0b_3x1')])
		with tf.variable_scope('branch_3'):
			branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
			branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

		net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
