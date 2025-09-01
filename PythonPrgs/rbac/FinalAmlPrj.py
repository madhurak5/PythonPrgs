import random
from warnings import simplefilter
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import adam_v2, rmsprop_v2
from keras.models import Sequential
import tensorflow as tf

from tensorflow.python.platform import flags
simplefilter(action='ignore', category=FutureWarning)

FLAGS = flags.FLAGS

fileName = "C://PythonPrgs/csvFiles/KDDTrain.csv"
data = pd.read_csv(fileName)
print(data.shape)
print(data.head())
# lab = LabelEncoder()
lab = OneHotEncoder()
# X_enc = pd.DataFrame(data)
# print(X_enc.head())
catCols = ['protocol_type','flag','service']
featArray = lab.fit_transform(data[catCols]).toarray()
print(featArray)
feature_labels = lab.categories_
print(feature_labels)
feature_labels1 = np.array(feature_labels)
print(feature_labels1)
feat2 = np.concatenate(feature_labels1)
print("Feat2 : ", feat2)
df = pd.DataFrame(featArray , columns=feat2)
print(df.head())
data2 = pd.concat([data, df], axis=1)
data2 = data2.drop(catCols, axis=1)
print(data2.shape)
print(data2.head())
lab1 = LabelEncoder()
data2['class'] = lab1.fit_transform(data2['class'])
y = data2['class']
data2 = data2.drop("class", axis = 1)
scaler = MinMaxScaler().fit(data2)
X_scaler = scaler.transform(data2)
# scaler = MinMaxScaler().fit(X_train)
# X_train_scaled = np.array(scaler.transform(X_train))
# X_test_scaled = np.array(scaler.transform(X_test))
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_scaler.shape[1],)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='softmax'))
# model.add(Dense(FLAGS.nb_classes, activation='softmax'))
# model.add(Dense())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("MLP ",model.summary())


X_train,X_test, y_train, y_test = train_test_split(X_scaler, y,test_size=0.20, random_state=41)

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
gnbModel = gnb.fit(X_train, y_train)
gnbPred = gnb.predict(X_test)
print("Naive Bayes : ------> ")
cm = confusion_matrix(y_test, gnbPred)
print("Confusion Matrix : \n",cm)
print("Accuracy Score", accuracy_score(y_test, gnbPred))
print("F-score ", f1_score(y_test, gnbPred))
print("Precision : ", precision_score(y_test, gnbPred))
print("Recall: ", recall_score(y_test, gnbPred))

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

tree = DecisionTreeClassifier()
treeModel = tree.fit(X_train, y_train)
treePred = tree.predict(X_test)
print("Decision Tree : ------> ")
cm = confusion_matrix(y_test, treePred)
print("Confusion Matrix : \n",cm)
print("Accuracy Score", accuracy_score(y_test, treePred))
print("F-score ", f1_score(y_test, treePred))
print("Precision : ", precision_score(y_test, treePred))
print("Recall: ", recall_score(y_test, treePred))

def getAttributes(dataset):
    return (dataset.columns)

def featSelectionKBest(noofFeat):
    import matplotlib.pyplot as plt
    from sklearn.feature_selection import SelectKBest, chi2
    bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    # plt.bar([i for i in range(len(fit.scores_))],fit.scores_)
    # plt.show()
    dfcolumns = pd.DataFrame(data2.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
    featureScores.columns = ['Specs','Score']   # naming the dataframe columns
    print(featureScores.nlargest(noofFeat,'Score').sort_values(by=['Score'], axis=0,ascending=False))  #print 10 best features
    return bestfeatures, dfscores, featureScores

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
for i in getAttributes(data2):
    if i in selColumns:
        newDf[[i]] = data2[[i]]


# X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)

bestfeatures, dfscores, featureScores = featSelectionKBest(10)
print("Best Features are: ", bestfeatures)
print(dfscores)
print(featureScores)