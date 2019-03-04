# -*- coding: utf-8 -*-
# @File    : conv_pooling.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/4
# @Desc    :

import tensorflow as tf

"""params:
filter(height,weight,current_depth,next_depth)
biases
"""
filter_weight = tf.get_variable(name='weights',shape=[5,5,3,16],initializer=tf.truncated_normal_initializer(mean=0.0,stddev=0.1))
biases = tf.get_variable(name='biases',shape=[16],initializer=tf.constant_initializer(value=0.1))

"""
tf.nn.conv2d
tf.nn.bias_add
tf.nn.max_pool
"""
conv = tf.nn.conv2d(input=input,filter=filter_weight,strides=[1,1,1,1],padding='SAME')
bias = tf.nn.bias_add(conv,biases)
actived_conv = tf.nn.relu(bias)
pool = tf.nn.max_pool(input=actived_conv,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
