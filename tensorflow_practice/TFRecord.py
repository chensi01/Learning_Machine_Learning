# -*- coding: utf-8 -*-
# @File    : TFRecord.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/9
# @Desc    : TFRecord 处理mnist数据

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets("datasets", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

pixels = images.shape[1]
num_examples = mnist.train.num_examples

filename = "datasets/mnist_tfrecord"
writer = tf.python_io.TFRecordWriter(filename)
for i in range(num_examples):
	image_raw = images[i].tostring()
	example = tf.train.Example(
		features=tf.train.Features(feature={'pixcels': _int64_feature(pixels), 'label': _int64_feature(np.argmax(labels[i])),
											'image_raw': _bytes_feature(image_raw)}))
	writer.write(example.SerializeToString())
writer.close()
