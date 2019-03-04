# -*- coding: utf-8 -*-
# @File    : mnist2_inference.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/1
# @Desc    :
import tensorflow as tf

"""1.定义网络结构相关的参数"""
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

"""2.定义获取变量的函数"""


def get_weight_variable(shape, regularizer):
	weights = tf.get_variable(name="weights", shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
	if regularizer != None:
		tf.add_to_collection("losses", regularizer(weights))
	return weights


"""3.定义网络的前向传播过程"""


def inference(input_tensor, regularizer):
	with tf.variable_scope("layer1"):
		weights = get_weight_variable(shape=[INPUT_NODE, LAYER1_NODE], regularizer=regularizer)
		biases = tf.get_variable(name="biases", shape=[LAYER1_NODE], initializer=tf.constant_initializer(value=0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
	with tf.variable_scope("layer2"):
		weights = get_weight_variable(shape=[LAYER1_NODE, OUTPUT_NODE], regularizer=regularizer)
		biases = tf.get_variable(name="biases", shape=[OUTPUT_NODE], initializer=tf.constant_initializer(value=0.0))
		layer2 = tf.matmul(layer1, weights) + biases
	return layer2
