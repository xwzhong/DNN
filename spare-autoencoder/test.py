#coding: utf-8
#date: 2016-06-29
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 0.1

import util
from bean import autoencoder,nn
import numpy as np

x = np.array([[0,0,1,0,0],
            [0,1,1,0,1],
            [1,0,0,0,1],
            [1,1,1,0,0],
            [0,1,0,1,0],
            [0,1,1,1,1],
            [0,1,0,0,1],
            [0,1,1,0,1],
            [1,1,1,1,0],
            [0,0,0,1,0]])
y = np.array([[0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [1],
            [0]])

######################   建立autoencoder   ######################
# 此处设为两层autoencoder（隐含层为2），每层分别包含3，2个node，第一个元素5为样本的维数
nodes=[5,800,20]
# 建立auto框架
ae = util.aebuilder(nodes)
# 训练，设置部分参数
ae = util.aetrain(ae, x, 6000)

######################   微调   ######################
# 建立完全体的autoencoder，数组[5,3,2,1]中
# 第一个元素表示样本的维度（输入层的node数）
# 最后一个元素（输出层的node数）为样本的预测值维度，
# 中间部分的元素为隐含层的node数
nodescomplete = np.array([5,800,20,1])
aecomplete = nn(nodescomplete)
# 将encoder后的隐含层添加到最终的网络中
for i in range(len(nodescomplete)-2):
    aecomplete.W[i] = ae.encoders[i].W[0]
aecomplete = util.nntrain(aecomplete, x, y, 10000)
print aecomplete.values[3]
print sum(abs(aecomplete.values[3]-y))/len(y)
