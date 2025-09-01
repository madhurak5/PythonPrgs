# import tensorflow as tf
# print(tf.__version__)
# h = tf.constant("Hello")
# w = tf.constant("World")
# hw = h + w
# with tf.Session() as sess:
#     ans = sess.run(hw)
# print("Answer : ", ans)
# import numpy as np
# import  pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# data = pd.read_csv("C://PythonPrgs/csvFiles/KDDTrain.csv")
# print(data.head())
# print(data.isna())
# print(data.dtypes[data.dtypes != 'int64'][data.dtypes != 'float64'])
# print(data['protocol_type'].unique())
# print(data['service'].unique())
# print(data['flag'].unique())
# print(data['class'].unique())
# # data[]
# import gym
# env =gym.make("Taxi-v3").env
# env.render()
# import os
# import sys
# def tree(x,i=0):
# 	ls=os.listdir(x)
# 	k=0
# 	print(' '*i,x)
# 	while k<len(ls):
# 		if '.' in ls[k]:
# 			print (' '*i,'|', '_'*2,ls[k])
# 		else:
# 			tree(x+'/'+ls[k],i+2)
# 		k=k+1
# tree("C://0Research")
import intake

intake.cat.states