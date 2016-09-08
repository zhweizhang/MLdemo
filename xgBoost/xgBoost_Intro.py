# -*- coding:utf-8 -*-

import xgboost as xgb
import numpy as np
import os

path_url = os.getcwd()

def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0 - p)
    return g, h

def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.0))) / len(y_hat)

if __name__ == "__main__":
    data_train = xgb.DMatrix(path_url + '\\agaricus_train.txt')
    data_test = xgb.DMatrix(path_url + "\\agaricus_test.txt")

    # set param
    # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'} #logitraw
    param = {'max_depth': 2, 'eta': 1, 'silent': 1}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 3
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    # bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)
    # error = sum(y != (y_hat > 0))   # together

    # error rate
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print y_hat
    print y
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print '样本总数:\t', len(y_hat)
    print '错误数目:\t%4d' % error
    print '错误率:\t%.2f%%' % (100*error_rate)