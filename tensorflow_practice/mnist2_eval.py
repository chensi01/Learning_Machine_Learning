# -*- coding: utf-8 -*-
# @File    : mnist2_eval.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/2
# @Desc    :

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist2_inference
import mnist2_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
	with tf.Graph().as_default() as g:
		"""1.定义输入输出格式"""
		x = tf.placeholder(dtype=tf.float32, shape=[None, mnist2_inference.INPUT_NODE], name='x-input')
		y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist2_inference.OUTPUT_NODE], name='y-input')
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
		"""2.用封装好的函数计算前向传播结果"""
		y = mnist2_inference.inference(x, None)
		"""3.计算正确率"""
		correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		"""4.用变量重命名的方式加载模型（滑动平均）"""
		variable_average = tf.train.ExponentialMovingAverage(mnist2_train.MOVING_AVERAGE_DECAY)
		variable_to_restore = variable_average.variables_to_restore()
		saver = tf.train.Saver(variable_to_restore)
		"""5.每隔n秒调用一次正确率的计算过程"""
		while True:
			with tf.Session() as sess:
				# (1)找到目录中最新模型的文件名
				ckpt = tf.train.get_checkpoint_state(mnist2_train.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					# (2)加载模型
					saver.restore(sess,ckpt.model_checkpoint_path)
					# (3)计算准确率
					global_step = ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]
					accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
					print("After %s training step(s),validation accuracy is %g" % (global_step, accuracy_score))
				else:
					print("No checkpoint file found")
					return
				time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
	mnist = input_data.read_data_sets("./datasets", one_hot=True)
	evaluate(mnist)
if __name__ == '__main__':
	tf.app.run()
