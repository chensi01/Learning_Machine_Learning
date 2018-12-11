"""
@FileName:day_11_knn.py

@Author：chensi_aria@foxmail.com

@Create date:2018/11/24

@description：机器学习100天

@Update date：2018/11/24

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def evaluate_metric(groundTruth, predLabel):
	c_m = metrics.confusion_matrix(groundTruth, predLabel)
	return [metrics.precision_score(groundTruth, predLabel, average='micro'),
			metrics.precision_score(groundTruth, predLabel, average='macro'),
			metrics.recall_score(groundTruth, predLabel, average='micro'),
			metrics.recall_score(groundTruth, predLabel, average='macro')]

# load data
data_set_path = os.path.abspath(os.path.join(os.getcwd(), '../../datasets/Social_Network_Ads.csv'))
data_set = pd.read_csv(data_set_path)
X = data_set.iloc[:, 2:4].values
Y = data_set.iloc[:, 4].values

# dara preprocessing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# train and test model
clf_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
clf_knn.fit(X_train,Y_train)
Y_pred = clf_knn.predict(X_test)

# evaluate and visualization
cm = metrics.confusion_matrix(Y_test, Y_pred)
print("confusion matrix : ",cm)
print(evaluate_metric(Y_test, Y_pred))
