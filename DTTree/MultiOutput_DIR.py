# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    N = 100
    x = np.random.rand(N)
    x.sort()
    y1 = np.sin(x) + np.random.randn(N) * 0.1
    y2 = np.cos(x) + np.random.randn(N) * 0.1
    y = np.vstack((y1, y2)).T
    x = x.reshape(-1, 1)

    deep = 10
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)

    x_test = np.linspace(-4, 4, num=100).reshape(-1, 1)
    y_hat = dt.predict(x_test)
    plt.scatter(y[:, 0], y[:, 1], c='r', s=40, label='Actual')
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='s', s=40, label='Depth=%d' % deep, alpha=0.7)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()