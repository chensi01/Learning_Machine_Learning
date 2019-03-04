# -*- coding: utf-8 -*-
# @File    : basic_example2.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/2/28
# @Desc    :
import tensorflow as tf

"""session"""
matrix1 = tf.constant(value=[[3, 3]], dtype=tf.float32)
matrix2 = tf.constant(value=[[2], [2]], dtype=tf.float32)
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
	r = sess.run(product)
	print(r)

"""Variable"""
counter_var = tf.Variable(initial_value=0, name='counter')
one = tf.constant(value=1)
add_op = tf.assign(counter_var, tf.add(counter_var, one))
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(5):
		sess.run(add_op)
		print(sess.run(counter_var))

"""placeholder"""
input1 = tf.placeholder(dtype=tf.float32, shape=None)
input2 = tf.placeholder(dtype=tf.float32, shape=None)
multi_op = tf.matmul(input1, input2)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(multi_op, feed_dict={input1: [[1,2]], input2: [[1],[2]]}))
