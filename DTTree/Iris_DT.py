# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]

iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'

if __name__ == "__main__":
    path = 'D:\pythonpro\MLDesign\Regression\iris.data'

    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4:iris_type})

    # 将数据的0-3列组成x,第4列得到y
    x, y = np.split(data, (4,), axis=1)

    x = x[:, :2]

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=30, min_samples_leaf=3)
    dt_clf = clf.fit(x, y)

    # 画图
    N, M = 500, 500
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max)
    t2 = np.linspace(x2_min, x2_max)
    x1, x2 = np.meshgrid(t1, t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)

    y_hat = dt_clf.predict(x_test)
    y_hat = y_hat.reshape(x1.shape)
    plt.pcolormesh(x1, x2, y_hat, cmap=plt.cm.Spectral, alpha=0.1) # 预测值的显示Paired/Spectral/coolwarm/summer/spring/OrRd/Orange
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', cmap=plt.cm.prism) # 样本显示
    plt.xlabel(iris_feature[0])
    plt.ylabel(iris_feature[1])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.show()

    # 训练集上的预测结果
    y_hat = dt_clf.predict(x)
    y = y.reshape(-1)
    print y_hat.shape
    print y.shape
    result = (y_hat == y)
    print y_hat
    print y
    print result
    c = np.count_nonzero(result)
    print c
    print 'Accuracy: %.2f%%' % (100 * float(c) / float(len(result)))



