# -*- coding: utf-8 -*-
# @File    : transfer_learning.py
# @Author  : chensi
# @Contact : chensi_aria@foxmail.com
# @Date    : 2019/3/8
# @Desc    : transfer_learning on ImageNet using Inception-v3


import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

"""
瓶颈层
"""
BOTTLE_NECK_TENSOR_SIZE = 2048
BOTTLE_NECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MODEL_DIR = 'model/'
MODEL_FILE = 'classify_image_graph_def.pb'
CACHE_DIR = 'tmp/'
INPUT_DATA = 'datasets/flower_data'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

"""
读取图像数据并划分训练/验证/测试集合
"""






"""
获取图像地址
"""





"""
获取图像被Inception-v3处理好的特征向量地址
"""

"""
使用Inception-v3处理图像，得到特征向量
"""


"""
获取图像的特征向量，找不到则使用Inception-v3处理得到特征向量并保存
"""



"""
选一个batch的图像作为训练数据
"""

"""
获取测试数据，计算正确率
、
"""








