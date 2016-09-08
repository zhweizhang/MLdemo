# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np
import scipy.sparse
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
import os

path_url = os.getcwd()

def read_data(path):
    y = []
    row = []
    col = []
    values = []
    r = 0      # 首行
    for d in open(path):
        d = d.strip().split()
        y.append(int(d[0]))
        d = d[1:]
        for c in d:
            key, value = c.split(':')
            row.append(r)
            col.append(int(key))
            values.append(float(value))
        r += 1
    x = scipy.sparse.csr_matrix((values, (row, col))).toarray()
    y = np.array(y)
    return x, y

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print acc
    print tip + '正确率:\t', float(acc.sum()) / a.size

if __name__ =="__main__":
    path = path_url + '\\agaricus_train.txt'
    x, y = read_data(path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    # Logistic
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic Regression')

    # XGBoost
    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 3, 'eta': 1, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 3}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost')