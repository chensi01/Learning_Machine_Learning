#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.7
@Author  : Chensi
@Explain :
@contact:  chensi_aria@foxmail.com
@contact:  sichen@stu.xmu.edu.cn
@Create date:2018/12/09
@FileName:day_11_logistic_regression_v2.py
@description：逻辑回归
@Update date：2018/12/09
@Software: PyCharm
@refernece :https://github.com/SSaishruthi/LogisticRegression_Vectorized_Implementation/blob/master/Logistic_Regression.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os


def weightInitialization(n_features):
	w = np.zeros((1, n_features))
	b = 0
	return w, b


def sigmoid_activation(result):
	final_result = 1 / (1 + np.exp(-result))
	return final_result


def model_optimize(w, b, X, Y):
	m = X.shape[0]

	# Prediction
	final_result = sigmoid_activation(np.dot(w, X.T) + b)
	Y_T = Y.T
	cost = (-1 / m) * (np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
	#

	# Gradient calculation
	dw = (1 / m) * (np.dot(X.T, (final_result - Y.T).T))
	db = (1 / m) * (np.sum(final_result - Y.T))

	grads = {"dw": dw, "db": db}

	return grads, cost


def model_predict(w, b, X, Y, learning_rate, no_iterations):
	costs = []
	for i in range(no_iterations):
		#
		grads, cost = model_optimize(w, b, X, Y)
		#
		dw = grads["dw"]
		db = grads["db"]
		# weight update
		w = w - (learning_rate * (dw.T))
		b = b - (learning_rate * db)
		#

		if (i % 100 == 0):
			costs.append(cost)
		# print("Cost after %i iteration is %f" %(i, cost))

	# final parameters
	coeff = {"w": w, "b": b}
	gradient = {"dw": dw, "db": db}

	return coeff, gradient, costs


def predict(final_pred, m):
	y_pred = np.zeros((1, m))
	for i in range(final_pred.shape[1]):
		if final_pred[0][i] > 0.5:
			y_pred[0][i] = 1
	return y_pred


file_path = os.path.abspath(os.path.join(os.getcwd(), '../datasets/iris-data.csv'))
df = pd.read_csv(file_path)
"""
看数据集信息
"""
# df.head()
# df.describe()
# df.info()
# Removing all null values row
df = df.dropna(subset=['petal_width_cm'])
# sns.pairplot(df, hue='class', height=2.5)
# df.info()
df['class'].replace(["Iris-setossa", "versicolor"], ["Iris-setosa", "Iris-versicolor"], inplace=True)
# df['class'].value_counts()
# print(df['class'].value_counts())
"""
simple logistic regression
"""
final_df = df[df['class'] != 'Iris-virginica']
# outlier
# sns.pairplot(final_df, hue='class', size=2.5)
final_df.hist(column='sepal_length_cm', bins=20, figsize=(10, 5))

# final_df.loc[final_df.sepal_length_cm < 1, ['sepal_length_cm']] = final_df['sepal_length_cm']*100
# final_df.hist(column = 'sepal_length_cm',bins=20, figsize=(10,5))
# label encoding
final_df['class'].replace(["Iris-setosa", "Iris-versicolor"], [1, 0], inplace=True)
inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0, 1, 2, 3]], axis=1)

#
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)
#
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)

X_tr_arr = X_train
X_ts_arr = X_test
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()
"""
自己实现
"""
#Get number of features
n_features = X_tr_arr.shape[1]
print('Number of Features', n_features)
w, b = weightInitialization(n_features)
#Gradient Descent
coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001,no_iterations=4500)
#Final prediction
w = coeff["w"]
b = coeff["b"]
print('Optimized weights', w)
print('Optimized intercept',b)
#
final_train_pred = sigmoid_activation(np.dot(w,X_tr_arr.T)+b)
final_test_pred = sigmoid_activation(np.dot(w,X_ts_arr.T)+b)
#
m_tr =  X_tr_arr.shape[0]
m_ts =  X_ts_arr.shape[0]
#
y_tr_pred = predict(final_train_pred, m_tr)
print('Training Accuracy',accuracy_score(y_tr_pred.T, y_tr_arr))
#
y_ts_pred = predict(final_test_pred, m_ts)
print('Test Accuracy',accuracy_score(y_ts_pred.T, y_ts_arr))

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()


"""
sklearn
"""
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_tr_arr, y_tr_arr)
pred = clf.predict(X_ts_arr)

print ('Accuracy from sk-learn: {0}'.format(clf.score(X_ts_arr, y_ts_arr)))


print("success")
