# 机器学习100天-Day3-多元线性回归

---
- 语言：python
- Reference：[Avik-Jain](https://github.com/Avik-Jain/100-Days-Of-ML-Code)  & [MLEveryday](https://github.com/MLEveryday/100-Days-Of-ML-Code) 

---

- step 1:   导入库和数据
- step 2：数据预处理
- step 3：训练模型
- step 4：测试模型
- step 5：可视化

![此处输入图片的描述][1]



```python
# -*- coding:utf-8 -*-

"""
@FileName:day_3_multiple_linear_regression.py

@Author：chensi_aria@foxmail.com

@Create date:2018/11/21

@description：机器学习100天

@Update date：2018/11/21

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
data_set_path = os.path.abspath(os.path.join(os.getcwd(),'../../datasets/50_Startups.csv'))
data_set = pd.read_csv(data_set_path)
X = data_set.iloc[:, :-1].values#.reshape(-1,1)#reshape(-1,1)将原数组变为1列
Y = data_set.iloc[:, 4].values#.reshape(-1,1)
#编码类别特征
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
#躲避虚拟变量陷阱
X = X[:,1:]
#填补缺失值
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=0, strategy='mean', axis=0)
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)#Imputer(missing_values="NAN", strategy="mean", axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)
#划分
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


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
plt.scatter([i for i in range(len(X_train))],Y_train,color='red')
plt.plot([i for i in range(len(X_train))],regressor.predict(X_train),color='blue')
plt.subplot(122)
plt.scatter([i for i in range(len(X_test))],Y_test,color='red')
plt.plot([i for i in range(len(X_test))],Y_pred,color='blue')
plt.show()
```


[1]: http://thyrsi.com/t6/613/1542778630x2890237508.jpg