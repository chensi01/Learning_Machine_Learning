# -*- coding:utf-8 -*-

"""
@FileName:day_2_simple_linear_regression.py

@Author：chensi_aria@foxmail.com

@Create date:2018/11/20

@description：机器学习100天

@Update date：2018/11/20

"""


"""
FrameWork

step 1:导入库和数据
step 2：数据预处理
step 3：训练模型
step 4：测试模型
step 5：可视化
"""




"""
step 1:导入库和数据
step 2：数据预处理
"""
#导入库和数据
import os
import pandas as pd
data_set_path = os.path.abspath(os.path.join(os.getcwd(),'../../datasets/studentscores.csv'))
# data_set_path = os.path.abspath(os.path.join(os.getcwd(),'../../datasets/test.csv'))
data_set = pd.read_csv(data_set_path)
X = data_set.iloc[:, 0].values.reshape(-1,1)#reshape(-1,1)将原数组变为1列
Y = data_set.iloc[:, 1].values.reshape(-1,1)
#填补缺失值
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)#Imputer(missing_values="NAN", strategy="mean", axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)
#划分
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split( X , Y , test_size = 0.25, random_state = 0)

"""
step 3：训练模型
step 4：测试模型
"""
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)


"""
step 5：可视化
"""
from matplotlib import pyplot as plt
plt.figure(12)
plt.subplot(121)
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.subplot(122)
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,Y_pred,color='blue')
plt.show()