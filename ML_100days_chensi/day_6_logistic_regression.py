"""
@FileName:day_6_logistic_regression.py

@Author：chensi_aria@foxmail.com

@Create date:2018/11/23

@description：机器学习100天

@Update date：2018/11/23

"""

"""
FrameWork

step 1:导入库和数据
step 2：数据预处理
step 3：训练和测试模型
step 4：评估和可视化
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def draw(x_set, y_set,classifier,title,xlabel,ylabel):
	X_set, y_set = x_set, y_set
	X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
						 np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
	plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
			 alpha=0.75, cmap=ListedColormap(('red', 'green')))
	plt.xlim(X1.min(), X1.max())
	plt.ylim(X2.min(), X2.max())
	for i, j in enumerate(np.unique(y_set)):
		plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],c=ListedColormap(('red', 'green'))(i), label=j)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.show()

# load data
data_set = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), '../datasets/Social_Network_Ads.csv')))
X = data_set.iloc[:, 2:4].values
Y = data_set.iloc[:, -1].values

# data preprocessing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25,
													random_state=0)  # random_state：随机数种子——其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。\
data_preprocessing = StandardScaler()
X_train = data_preprocessing.fit_transform(X_train)
X_test = data_preprocessing.transform(X_test)

# build model and test
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

# evlaluating and visualization
c_m = confusion_matrix(Y_test, Y_pred)
draw(X_train, Y_train,classifier,"LOGISTIC(Training set)","Age","Estimated Salary")
draw(X_test, Y_test,classifier,"LOGISTIC(Test set)","Age","Estimated Salary")