# -*- coding: utf-8 -*-
# @File    : minist.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/2/28
# @Desc    : 手写数字识别

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# MNIST数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99


# 辅助函数，计算前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
	if avg_class == None:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
		return tf.matmul(layer1, weights2) + biases2
	else:
		layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
		return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 定义训练模型的函数
def train(mnist):
	# 1.定义接受数据的结点
	x = tf.placeholder(dtype=tf.float32, shape=[None, INPUT_NODE], name='x-input')
	y_ = tf.placeholder(dtype=tf.float32, shape=[None, OUTPUT_NODE], name='y-input')
	# 2.定义网络模型中的参数
	weights1 = tf.Variable(tf.truncated_normal(shape=[INPUT_NODE, LAYER1_NODE], mean=0.0, stddev=0.1))
	biases1 = tf.Variable(tf.constant(value=0.1, shape=[LAYER1_NODE]))
	weights2 = tf.Variable(tf.truncated_normal(shape=[LAYER1_NODE, OUTPUT_NODE], mean=0.0, stddev=0.1))
	biases2 = tf.Variable(tf.constant(value=0.1, shape=[OUTPUT_NODE]))
	# 3.定义存储轮数（不可训练）
	global_step = tf.Variable(initial_value=0, trainable=False)
	# 4.初始化滑动平均类
	variables_averages = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
	variables_averages_op = variables_averages.apply(tf.trainable_variables())
	# 5.计算前向传播结果
	y = inference(x, None, weights1, biases1, weights2, biases2)
	average_y = inference(x, variables_averages, weights1, biases1, weights2, biases2)
	# 6.计算损失函数
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	regularization = regularizer(weights1) + regularizer(weights2)
	loss = cross_entropy_mean + regularization
	# 7.设置指数衰减的学习率
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
											   LEARNING_RATE_DECAY)

	# 8.优化损失函数
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	# 9.更新参数及其滑动平均值
	train_op = tf.group(train_step, variables_averages_op)
	# 10.计算正确率
	correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


	# 初始化会话并开始训练
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		validation_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		test_feed = {x: mnist.test.images, y_: mnist.test.labels}
		for i in range(TRAINING_STEPS):
			if i%1000 == 0:
				validation_acc = sess.run(accuracy,feed_dict=validation_feed)
				print("After %d training step(s),validation accuracy is %g"%(i,validation_acc))
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op,feed_dict={x:xs,y_:ys})
		test_acc = sess.run(accuracy,feed_dict=test_feed)
		print("After %d training step(s),test accuracy is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
	mnist = input_data.read_data_sets("./datasets", one_hot=True)
	train(mnist)
if __name__ == '__main__':
	tf.app.run()


# # mnist数据集
# print("Training data size", mnist.train.num_examples)
# print("Validation data size", mnist.validation.num_examples)
# print("Testing data size", mnist.test.num_examples)
# print("Example training data:", mnist.train.images[0])
# print("Example training data label:", mnist.train.labels[0])
# # 从训练数据中读取一部分做欸一个训练batch
# batch_size = 100
# xs,ys = mnist.train.next_batch(batch_size)
# print("X shape:",xs.shape)
# print("Y shape:",ys.shape)
