# -*- coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D
import math
from PIL import Image

path_url = os.getcwd()

def rebuild(U, Sigma, V, num):
    lenth = len(U)
    width = len(V[0])
    pic = np.zeros((lenth, width))
    for k in range(num):
        for kk in range(lenth):
            pic[kk] += Sigma[k] * U[kk][k] * V[k]
    return pic.astype('uint8')

if __name__ == "__main__":

    # 心形线
    # theta = np.linspace(0, 2 * math.pi, 1000)
    # f_theta = np.sin(theta) * np.sqrt(np.abs(np.cos(theta))) / (np.sin(theta) + 7 / 5) - 2 * np.sin(theta) + 2
    # y = f_theta * np.sin(theta)
    # x = f_theta * np.cos(theta)
    # plt.plot(x, y)
    # plt.title('xinxingxian')
    # plt.text(-2, -1, '$f(\\theta) = \\frac{sin(\\theta)(|cos(\\theta)|)^{0.5}}\
    #     {sin(\\theta)+1.4}-2sin(\\theta)+2$', fontsize=16)
    # plt.show()

    #SVD
    img = Image.open(path_url + '\Koala.jpg')
    plt.figure("koala")
    plt.imshow(img)

    #转化为灰度
    gray = img.convert('L')
    plt.figure("gray")
    plt.show(gray)

    U,Sigma,V = np.linalg.svd(gray, full_matrices=False)
    for num in range(0, 200, 20):
        b = rebuild(U,Sigma,V,num)
        Image.fromarray(b).save(path_url + '\svd' + str(num) + '.jpg')

    #分离三通道
    r,g,b=img.split()
    U1,Sigma1,V1 = np.linalg.svd(r,full_matrices=False)
    U2,Sigma2,V2 = np.linalg.svd(g,full_matrices=False)
    U3,Sigma3,V3 = np.linalg.svd(b,full_matrices=False)
    for num in range(0,200,20):
        r1 = rebuild(U1,Sigma1,V1,num)
        g1 = rebuild(U2,Sigma2,V2,num)
        b1 = rebuild(U3,Sigma3,V3,num)
        I = np.stack((r1,g1,b1),2)
        Image.fromarray(I).save(path_url + '\svdrgb' + str(num) + '.jpg')