import logging
import random
from warnings import simplefilter

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

simplefilter(action='ignore', category=FutureWarning)


def getData(fName):
    fileName = "C://PythonPrgs/csvFiles/"+fName+".csv"
    data = pd.read_csv(fileName)
    return data

def prnDsDetails(dataset):
    print(dataset.shape)
    print(dataset.head())

def getAttributes(dataset):
    return (dataset.columns)
#
# def modelFitPredict(modelNB, X_train, y_train, X_test):
#     model = modelNB.fit(X_train, y_train)
#     pred  = modelNB.predict(X_test)
#     return model, pred

# def modelScores(y_test, pred):
#     print("Confusion Matrix \n", confusion_matrix(y_test, pred))
#     print("Accuracy Score", accuracy_score(y_test,pred))
#     print("F-score ", f1_score(y_test, pred))
#     print("Precision : ", precision_score(y_test, pred))
#     print("Recall: ", recall_score(y_test, pred))
#
#
# def naiveBayesAlgo():
#     from sklearn.naive_bayes import BernoulliNB
#     gnb = BernoulliNB()
#     gnbModel = gnb.fit(X_train, y_train)
#     gnbPred = gnb.predict(X_test)
#     print("Naive Bayes : ",accuracy_score(y_test, gnbPred))
#     return modelScores(y_test, gnbPred)

def featSelectionKBest(noofFeat):
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import SelectKBest, chi2
    bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    # plt.bar([i for i in range(len(fit.scores_))],fit.scores_)
    # plt.show()
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
    featureScores.columns = ['Specs','Score']   # naming the dataframe columns
    print(featureScores.nlargest(noofFeat,'Score').sort_values(by=['Score'], axis=0,ascending=False))  #print 10 best features
    return bestfeatures, dfscores, featureScores

def stdscaling(dataset):
    # scaler = StandardScaler().fit(dataset)
    scaler = MinMaxScaler().fit(dataset)
    X_scaler = scaler.transform(dataset)
    return X_scaler

def featureSelection(X_train, y_train):
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=0)
    rfe =RFE(clf, n_features_to_select=1)
    rfe.fit(X_train, y_train)
    print("Ranked features")
    colNames = d1.columns
    newCols = list (colNames) # [i] for i in d1.columns)
    print(sorted(zip(map(lambda x : round (x, 4), rfe.ranking_), newCols)))

d1 = getData("KDDTrain")
d2 = getData("KDDTest")
d1 = d1.replace(r'^\s*$', np.nan, regex=True)
lab = LabelEncoder()
lab1 = OneHotEncoder()
X_enc = pd.DataFrame(d1)
print(X_enc.head())
catCols = ['protocol_type','flag','service','class']  # 'AttCat','attType','role',
dsCols = d1[catCols]

uni_Protocol = dsCols['protocol_type'].unique()
str1 = "Protocol_type_"
uni_Protocol2 = [str1+x for x in uni_Protocol]
dummyCols = uni_Protocol2
print("Dummy Columns")
print(dummyCols)
for i in catCols:
    X_enc[i] = lab.fit_transform(X_enc[i])


y = X_enc.loc[:,'class']
y1 = X_enc.loc[:,'class']
X_enc = X_enc.drop(['class'], axis=1)
X = X_enc

noofFeat = 10
print("Hello Reduced features ")
bestfeatures, dfscores , featureScores = featSelectionKBest(noofFeat)
feats = featureScores.loc[:15,'Specs']
print("Features only : \n", feats)
print("Best Features : \n", bestfeatures)
selBestFeatures = featureScores.nlargest(noofFeat,'Score')
print("Selected Bst Features : \n", selBestFeatures.iloc[:noofFeat,0])
selBestColumns = selBestFeatures.iloc[:,0].values
# print(selColumns)

k=0
selColumns = []
# myCols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','num_failed_logins','logged_in','root_shell','su_attempted','num_root','num_file_creations','num_access_files','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','same_srv_rate','diff_srv_rate','role','class']
for i in selBestFeatures.iloc[:, 0]:
    selColumns.append(i)

newDf = pd.DataFrame()
for i in getAttributes(X_enc):
    if i in selColumns:
        newDf[[i]] = X_enc[[i]]


X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
gnbModel = gnb.fit(X_train, y_train)
gnbPred = gnb.predict(X_test)
print("Confusion Matrix \n", confusion_matrix(y_test, gnbPred))
print("F-score ", f1_score(y_test, gnbPred))
print("Precision : ", precision_score(y_test, gnbPred))
print("Recall: ", recall_score(y_test, gnbPred))
print("Naive Bayes : ",accuracy_score(y_test, gnbPred))

bestfeatures, dfscores, featureScores = featSelectionKBest(10)
print("Best Features are: ", bestfeatures)
print(dfscores)
print(featureScores)

labeldf=d1['class']
labeldf_test=d2['class']
# change the label column
# newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
#                            'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
#                            ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
#                            'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf=labeldf.replace({ 'normal' : 0, 'anomaly' : 1 })
# newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
#                            'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
#                            ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
#                            'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
d1['class'] = newlabeldf
# from advertorch.attacks import FGSM
# FGSM.perturb()

to_drop_normal = [0]
to_drop_anomaly = [1]
X_normalDf = d1[~d1['class'].isin(to_drop_normal)]
y_normalDf = X_normalDf['class']

labEnd = LabelEncoder()
X_enc_nor = pd.DataFrame(X_normalDf)
print(X_enc_nor.head())
catCols_nor = ['protocol_type','flag','service','class']  # 'AttCat','attType','role',
dsCols_nor = X_normalDf[catCols_nor]

uni_Protocol = dsCols_nor['protocol_type'].unique()
str1 = "Protocol_type_"
uni_Protocol2 = [str1+x for x in uni_Protocol]
dummyCols = uni_Protocol2
print("Dummy Columns Nor")
print(dummyCols)
for i in catCols:
    X_enc_nor[i] = lab.fit_transform(X_enc_nor[i])


y = X_enc_nor.loc[:,'class']
y1 = X_enc_nor.loc[:,'class']
X_enc_nor = X_enc_nor.drop(['class'], axis=1)
X = X_enc_nor
X_normalDf = X_normalDf.drop(['class'], axis = 1 )
print("X_normalDf shape", X_normalDf.shape)
print("y_normalDf  shape ", y_normalDf.shape)
print(X_normalDf.columns)
# y_normalDf = d1['class']
X_train_nor, X_test_nor, y_train_nor, y_test_nor = train_test_split(X_enc_nor, y1, test_size=0.20, random_state=41)
print("Shape of dataframe for normal records: ", X_normalDf.shape)
anomalyDf = d1[~d1['class'].isin(to_drop_anomaly)]
print("Shape of dataframe for anomaly records: ", anomalyDf.shape)
# d2['class'] = newlabeldf_test
print("d1 classes")
print(d1['class'].head(10))
featureSelection(X_train, y_train)
print("X_train shape", X_train.shape)
print("Y train shape ", y_train.shape)
print("Feature Selection (Normal): ")
featureSelection(X_normalDf, y_normalDf)



X_anomalyDf = d1[~d1['class'].isin(to_drop_anomaly)]
y_anomalyDf = X_anomalyDf['class']

labEnd = LabelEncoder()
X_enc_ano = pd.DataFrame(X_anomalyDf)
print(X_enc_ano.head())
catCols_ano = ['protocol_type','flag','service','class']  # 'AttCat','attType','role',
dsCols_ano = X_anomalyDf[catCols_ano]

uni_Protocol = dsCols_ano['protocol_type'].unique()
str1 = "Protocol_type_"
uni_Protocol2 = [str1+x for x in uni_Protocol]
dummyCols = uni_Protocol2
print("Dummy Columns Anomaly")
print(dummyCols)
for i in catCols:
    X_enc_ano[i] = lab.fit_transform(X_enc_ano[i])


y = X_enc_ano.loc[:,'class']
y2 = X_enc_ano.loc[:,'class']
X_enc_ano = X_enc_ano.drop(['class'], axis=1)
X = X_enc_ano
X_anomalyDf = X_anomalyDf.drop(['class'], axis = 1 )
print("X_anomalyDf shape", X_anomalyDf.shape)
print("y_anomalyDf  shape ", y_anomalyDf.shape)
print(X_normalDf.columns)
# y_normalDf = d1['class']
X_train_nor, X_test_nor, y_train_nor, y_test_nor = train_test_split(X_enc_ano, y2, test_size=0.20, random_state=41)
featureSelection(X_anomalyDf, y_anomalyDf)

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
gnbModel = gnb.fit(X_train_nor, y_train_nor)
gnbPred = gnb.predict(X_test_nor)
print("Confusion Matrix \n", confusion_matrix(y_test_nor, gnbPred))
print("F-score ", f1_score(y_test_nor, gnbPred))
print("Precision : ", precision_score(y_test_nor, gnbPred))
print("Recall: ", recall_score(y_test_nor, gnbPred))
print("Naive Bayes : ",accuracy_score(y_test_nor, gnbPred))
# mod1, pred = modelFitPredict(modelNB, X_train, y_train, X_test)
# print("Model : ", mod1)
print("Predictions: ", gnbPred)
# modelScores(y_test, pred)

# from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
# fast_gradient_method()
# from setuptools import find_packages
# prj = find_packages()
# print(prj)