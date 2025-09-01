# import numpy as np
# import pandas as pd
# import seaborn as sns
# a = [11,22, 31,5, 6, 9, 89, 12,54,24]
# k = int(input(("Enter value of k: ")))
# arrLen = len(a)
# print(arrLen)
# print(k)
# # pre = []
# mandatory = int(input(("Enter no of mandatory attributes: ")))
# mand = [None]*mandatory
# for p in range (0, mandatory):
#     mand[p] = int(input())
# koutofn = int(input(("Enter how many values should match : ")))
# if (k > arrLen):
#     print("Invalid")
# else:
#     print("Done")
#     print("Enter the values to be matched:")
#     b = [None] * k
#     c = [None] * k
#     pre = [None] * k
#     arr = []
#     for p in range(0, k):
#         b[p] = int(input())
#
#     for i in range(0, k):
#         m = b[i]
#         for j in range(0, arrLen):
#             if  a[j] == m:
#                 c[i] = 1
#             else:
#                 pass
#     cnt = 0
#     for p in range(0, k):
#         ct = c[p]
#         if ct == 1:
#             cnt = cnt + 1
#     for i in range (0, mandatory):
#         mat_mand = mand[i]
#         for j in range(0, arrLen):
#             if mat_mand == a[j]:
#                 pre[i] = 1
#             else:
#                 pass
#     cntm = 0
#     for p in range(0, mandatory):
#         ctm = pre[p]
#         if ctm == 1:
#             cntm = cntm + 1
#     print("pre", pre)
#     print("cntm", cntm)
# if cnt >= koutofn and cntm == mandatory:
#     print("access granted")
# else:
#     print("access denied")



import numpy as np
import pandas as pd
from  kmodes.kmodes import KModes
from setuptools import find_packages

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, classification_report, matthews_corrcoef, confusion_matrix
from sklearn.decomposition import PCA
import  seaborn as sns
import random
import seaborn as sns
from mlxtend.plotting import plot_pca_correlation_graph
import sklearn.metrics
# from warnings import simplefilter
# simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

from sklearn import metrics
from scipy.stats import zscore
import os

dsNew = pd.read_csv("C://PythonPrgs/csvFiles/Files/newFile24AttCatRoleClass.csv")
def minmaxscaling(dataset):
    minMaxScaler = MinMaxScaler().fit(dataset)
    X_scaler = minMaxScaler.transform(dataset)
    return X_scaler

def naiveBayesAlgo():
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
    gnb = GaussianNB()
    gnbModel = gnb.fit(X_train, y_train)
    gnbPred = gnb.predict(X_test)
    print("----------------- Naive Bayes -----------------")
    acc = accuracy_score(y_test, gnbPred)
    misclass = 1 - acc
    print("Accuracy Score : ", acc)
    print("Precision Score : ", precision_score(y_test, gnbPred, average='macro'))
    print("Recall Score : ", recall_score(y_test, gnbPred, average='macro'))
    print("F-score : ", f1_score(y_test, gnbPred, average='macro'))
    cmg = confusion_matrix(y_test, gnbPred)
    print("Confusion Matrix : \n", cmg)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    sns.heatmap(cmg, annot=True)
    plt.show()

def LogReg():
    from  sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
    logReg = LogisticRegression()
    logRegModel = logReg.fit(X_train, y_train)
    logPred = logReg.predict(X_test)
    print("----------------- Logistic Regression -----------------")
    acc = accuracy_score(y_test, logPred)
    misclass = 1 - acc
    print("Accuracy Score : ", acc)
    print("Precision Score : ", precision_score(y_test, logPred, average='macro'))
    print("Recall Score : ", recall_score(y_test, logPred, average='macro'))
    print("F-score : ", f1_score(y_test, logPred, average='macro'))
    cmg = confusion_matrix(y_test, logPred)
    print("Confusion Matrix : \n", cmg)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    sns.heatmap(cmg, annot=True)
lab = LabelEncoder()


X_enc = pd.DataFrame(dsNew)
catCols = ['protocol_type','flag','service','AttCat','attType','role','class']
for i in catCols:
    X_enc[i] = lab.fit_transform(X_enc[i])
# print()
y = X_enc.loc[:,'class']
X_enc = X_enc.drop(['id1','id','class', 'no','AttCat'], axis=1)
X = X_enc
print("With all features ...............")
X_scaler = minmaxscaling(X_enc)
X = X_scaler
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# print("X_train shape 1 : ", X_train.shape)

naiveBayesAlgo()
LogReg()

from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
noOfMandFeat = 10
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=noOfMandFeat)
rfe1 = rfe.fit(X_train, y_train)
selectedFeatures = pd.DataFrame({'Feature ' : list(X_enc.columns), 'Ranking' : rfe1.ranking_})
print(selectedFeatures.sort_values(by='Ranking'))
featSort = [None] * len(selectedFeatures)
featSort = selectedFeatures.sort_values(by='Ranking')
print("Sorted -> ", featSort)
print("--------")
mand = [None] * noOfMandFeat
# print(featSort.loc[0][0])
j = 0
for i in range(0, featSort.shape[0]):
    if (rfe1.ranking_[i] == 1):
        print(selectedFeatures.loc[i][0])
        mand[j] = selectedFeatures.loc[i][0]
        j = j +1
    else:
        pass
print("mand->")
print(mand)
koutofn = [None] * 5
j = 0
for i in range(0, featSort.shape[0]):
    if (rfe1.ranking_[i] in range(2, 7)):
        koutofn[j] = selectedFeatures.loc[i][0]
        j = j + 1
    else:
        pass
print("Non-mandatory -> ", koutofn)
#
newData = pd.DataFrame(dsNew, columns=mand)
# newData['class'] = dsNew['class']
# print(newData.head())
#
# #
# # #

X_enc_new = pd.DataFrame(newData)
y = newData['role']
# # # catCols = ['protocol_type','flag','service','AttCat','attType','role','class']
# #
# # # X_enc_new[i] = lab.fit_transform(X_enc_new[i])
# # # print()
# y = X_enc_new.loc[:,'class']
# X_enc = X_enc_new.drop(['class'], axis=1)
# X = X_enc_new
# print("With RFE features...............")
# X_scaler = minmaxscaling(X_enc_new)
# X = X_scaler
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# # print("X_train shape 1 : ", X_train.shape)
#
# naiveBayesAlgo()
# LogReg()
# from sklearn.model_selection import RepeatedStratifiedKFold
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
# score = cross_val_score(mod1, X, y, scoring = 'accuracy', cv = cv, n_jobs=-1)
# print(score)
# cor = dsNew.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
# print("Mandatory attributes with correlation > 90")
#
# for i in cor:
#     print(i)
#
# newDatak = pd.DataFrame(dsNew, columns=koutofn)
# newDatak['class'] = dsNew['class']
# print(newData.head())
#
# #
# # #
# X_enc_newk = pd.DataFrame(newDatak)
# # # catCols = ['protocol_type','flag','service','AttCat','attType','role','class']
# #
# # # X_enc_new[i] = lab.fit_transform(X_enc_new[i])
# # # print()
# yk = X_enc_newk.loc[:,'class']
# X_enck = X_enc_newk.drop(['class'], axis=1)
# X = X_enc_newk
# print("With RFE features.. k out of n.............")
# X_scalerk = minmaxscaling(X_enc_newk)
# Xk = X_scalerk
# X_train, X_test, y_train, y_test = train_test_split(Xk, yk, test_size=0.20, random_state=41)
# # print("X_train shape 1 : ", X_train.shape)
#
# naiveBayesAlgo()
# LogReg()