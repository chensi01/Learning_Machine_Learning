# 机器学习100天-Day1-数据预处理

---
- 语言：python
- Reference：[Avik-Jain](https://github.com/Avik-Jain/100-Days-Of-ML-Code)  & [MLEveryday](https://github.com/MLEveryday/100-Days-Of-ML-Code) & [blog](https://www.cnblogs.com/jasonfreak/p/5448385.html)

---

## step 1:导入需要的库
- NumPy：包含数学计算函数
- Pandas：用来导入和管理数据集

## step 2:导入数据集
- 数据集通常是.csv格式
- Pandas的read_csv方法可以将本地csv读为数据帧

## step 3:处理缺失数据
- 原因：为了不降低模型性能
- 方法：使用均值或中间值代替缺失数据
- sklearn.preprocessing中的imputer类

## step 4:将类别标签解析为数字
- 原因：标签值不能进行数学计算
- sklearn.preprocessing中的LabelEncoder类

## step 5:数据集分为训练/测试集
- 训练集训练模型，测试集测试模型表现
- sklearn.model_selection中的train_test_split()方法(原文的sklearn.crossvalidation不行） 

## step 6:特征缩放
- sklearn.preprocessing中的StandardScalar类





```python
    # -*- coding:utf-8 -*-

"""
@FileName:day_1_data_preprocessing.py

@Author：chensi_aria@foxmail.com

@Create date:2018/11/19

@description：机器学习100天

@Update date：2018/11/20

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
step-1 read csv
"""
current_path = os.getcwd()
data_set_dir = os.path.abspath(os.path.join(os.getcwd(), "../../datasets"))
data_set_path = os.path.abspath(os.path.join(data_set_dir,"Data.csv"))
data_set = pd.read_csv(data_set_path)
X = data_set.iloc[ : , :-1].values	#根据标签的所在位置，从0开始计数，选取列&0:n则第只取到n-1；最后一列用-1表示
Y = data_set.iloc[ : , 3].values


"""
step-2 处理缺失值
"""
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#策略：mean(均值) median（中位数） most_frequent(众数)
#axis：指定轴数，默认axis=0代表列，axis=1代表行
imputer = imputer.fit(X[ : , 1:3])	#拟合数据
X[ : , 1:3] = imputer.transform(X[ : , 1:3])	#转换数据



"""
step-3 将类别属性解析为数字
"""

labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
#sklearn.preprocessing.OneHotEncoder(n_values=’auto’, categorical_features=’all’,dtype=<type ‘numpy.float64’>, sparse=True,handle_unknown=’error’)
#categorical_features：可能取值为all(所有的特征都被视为分类特征)、indices数组(示分类特征的indices值)或mask(特征长度数组)
#sparse：若为True时，返回稀疏矩阵，否则返回数组
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)


"""
step-4 拆分训练集和测试集
"""

X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
#test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量;
# random_state：是随机数的种子。

"""
step-5 特征缩放
"""
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
#fit:找到\mu和\sigma
#transform:x = x-\mu/\sigma
#StandardScaler的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据。即只fit训练数据，测试数据做transform。这里测试集也fit了。
print (X_train)
print (Y_train)
```
