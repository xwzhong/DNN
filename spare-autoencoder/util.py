#coding: utf-8
#date: 2016-06-29
#mail: artorius.mailbox@qq.com
#author: xinwangzhong -version 0.1

import sys
import numpy as np
from bean import nn,autoencoder

# 激活函数
def sigmod(x):
	return 1.0 / (1.0 + np.exp(-x))

#前馈函数
def nnff(nn,x,y):
	layers = nn.layers
	numbers = x.shape[0]
	# 赋予初值
	nn.values[0] = x
	for i in range(1,layers):
		nn.values[i] = sigmod(np.dot(nn.values[i-1],nn.W[i-1])+nn.B[i-1])

	# 初始化P值
	for j in range(1,layers-1):
		nn.P[j] = nn.values[j].sum(axis = 0)/(nn.values[j].shape[0])
	# 计算KL项(增加)
	sparsity = nn.sparse*np.log(nn.sparse/nn.P[layers-2])+(1-nn.sparse)*np.log((1-nn.sparse)/(1-nn.P[layers-2]))
	# nn.error：输出层中实际值与预测值的差值
	nn.error = y - nn.values[layers-1]
	nn.loss = 1.0/2.0*(nn.error**2).sum()/numbers
	return nn

#BP函数
def nnbp(nn, spare_flag):
	layers = nn.layers;
	
	#初始化delta
	deltas = list();
	for i in range(layers):
		deltas.append(0)
	
	#最后一层的delta为(sigmoid函数的导数为g(x)'=g(x)*(1-g(x)):
	deltas[layers-1] = -nn.error*nn.values[layers-1]*(1-nn.values[layers-1])

	#其他层的delta为
	for j in range(layers-2, 0, -1):# 从后往前计算
		if spare_flag:
			pj = np.ones([nn.values[j].shape[0],1])*nn.P[j]
			sparsity = nn.beta*(-nn.sparse/pj+(1-nn.sparse)/(1-pj))
			deltas[j] = (np.dot(deltas[j+1],nn.W[j].T)+sparsity)*nn.values[j]*(1-nn.values[j])
		else:
			deltas[j] = np.dot(deltas[j+1], nn.W[j].T)*nn.values[j]*(1-nn.values[j])
		
	#更新W值
	for k in range(layers-1):
		# print (deltas[k+1].shape[0])
		# sys.exit()
		nn.W[k] -= nn.u*np.dot(nn.values[k].T,deltas[k+1])/(deltas[k+1].shape[0])
		nn.B[k] -= nn.u*deltas[k+1]/(deltas[k+1].shape[0])
	return nn

#对神经网络进行训练
def nntrain(nn, x, y, iterations, spare=False):
	for i in range(iterations):
		nnff(nn,x,y)
		nnbp(nn, spare)
	return nn

#建立autoencoder框架
def aebuilder(nodes):
	# layers：第一个层的node数与每个样本的维度相等
	layers = len(nodes)
	ae = autoencoder()
	for i in range(layers-1):
		ae.add_one(nn([nodes[i], nodes[i+1], nodes[i]]))
	return ae

#训练autoencoder
def aetrain(ae,x,interations):
	elayers = len(ae.encoders)
	for i in range(elayers):
		#单层训练
		ae.encoders[i] = nntrain(ae.encoders[i], x, x, interations, spare=True)
		#单层训练后，获取该层中间层的值，作为下一层的训练
		nntemp = nnff(ae.encoders[i], x, x)
		x = nntemp.values[1]   
	return ae
