# -*- coding: utf-8 -*-
# @File    : mnist2_train.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/1
# @Desc    :

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist2_inference

# 配置训练网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的文件名与路径
MODEL_SAVE_PATH = "./model"
MODEL_NAME = "model.ckpt"


def train(mnist):
	"""1.定义输入输出的占位符"""
	x = tf.placeholder(dtype=tf.float32, shape=[None, mnist2_inference.INPUT_NODE], name='x_input')
	y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist2_inference.OUTPUT_NODE], name='y_input')
	global_step = tf.Variable(initial_value=0, trainable=False)
	"""2.定义正则化函数，计算前向传播结果"""
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	y = mnist2_inference.inference(x, regularizer)
	"""3.定义损失函数/学习率/滑动平均/训练过程"""
	variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variable_average_op = variable_average.apply(tf.trainable_variables())
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
	learning_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE, global_step=global_step,
											   decay_steps=mnist.train.num_examples / BATCH_SIZE,
											   decay_rate=LEARNING_RATE_DECAY)
	train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss=loss,
																						 global_step=global_step)
	train_op = tf.group(train_step, variable_average_op)
	saver = tf.train.Saver()
	"""4.训练"""
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		for i in range(TRAINING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			if i % 1000 == 0:
				print("After %d training step(s),loss on training batch is %g" % (i, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
	mnist = input_data.read_data_sets("./datasets", one_hot=True)
	train(mnist)


if __name__ == '__main__':
	tf.app.run()
