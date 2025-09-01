import collections.abc

import advertorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import warnings
from advertorch.attacks import GradientSignAttack
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
data = pd.read_csv("C://PythonPrgs/csvFiles/KDDTest.csv")
catCols = ['protocol_type', 'service', 'flag', 'class']
ohe = OneHotEncoder()
feature_array = ohe.fit_transform(data[catCols]).toarray()
print(feature_array)
feature_labels = ohe.categories_
print(feature_labels)
feature_labels1 = np.array(feature_labels)
print(feature_labels1)
# from iteration_utilities import flatten
feat2 = np.concatenate(feature_labels1)
print("Feat2 : ", feat2)
# print("Feature Labels : ", feature_labels)
# df = pd.DataFrame(data, columns=feature_labels)
# print(df)
df = pd.DataFrame(feature_array , columns=feat2)
print(df.head())

print(df.shape)
data2 = pd.concat([data, df], axis=1)
print(data2.shape)
data2 = data2.drop(['protocol_type', 'service', 'flag'], axis=1)
print(data2.shape)
cols = data2.columns
print(cols)
X = data2.drop("class", axis=1)
y = data2["class"]
scaler = MinMaxScaler()
X_scaler = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=42)

gnb =GaussianNB()
gnbModel = gnb.fit(X_train, y_train)
gnbPred = gnb.predict(X_test)
print("Naive Bayes : ",accuracy_score(y_test, gnbPred))
cm = confusion_matrix(y_test, gnbPred)
print(cm)
print("Recall score ", recall_score(y_test, gnbPred))
import torchvision

# def zero_gradients(x):
#     if isinstance(x, torch.Tensor):
#         if x.grad is not None:
#             x.grad.detach_()
#             x.grad.zero_()
#         elif isinstance(x, collections.abc.Iterable):
#             for elem in x:
#                 zero_gradients(elem)

# from torch.autograd.gradcheck import zero_gradients
#
# adv_eg = advertorch.attacks.GradientSignAttack(predict=gnbPred, loss_fn=None, eps=0.015, clip_min=0.0, clip_max=1.0, targeted=False)
# b = adv_eg.perturb(X_train, y_train)
