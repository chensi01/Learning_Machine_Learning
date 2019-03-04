# -*- coding: utf-8 -*-
# @File    : basic_example.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/2/28
# @Desc    : basic examples of tensorflow

import tensorflow as tf
import numpy as np


# 想学习y=0.1x+0.3，先人工生成训练数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.3 + 0.1

# 定义网络模型中的参数
weight = tf.Variable(initial_value=tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0))
biases = tf.Variable(initial_value=tf.zeros(shape=[1]))
# 定义前向传播的输出
y_ = weight * x_data + biases
#定义loss
loss = tf.reduce_mean(tf.square(y_-y_data))
#定义梯度下降优化器，最小化loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

#创建session，初始化参数
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#训练
for step in range(200):
	sess.run(train_op)
	if step%20 == 0:
		w_,b_,loss_ = sess.run(weight),sess.run(biases),sess.run(loss)
		print("After %d training step(s),loss is %g."%(step,loss_))
		print(w_,b_)
