from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import tensorflow as tf
import keras
from keras.layers import Dense, Dropout
from keras.optimizers import adam_v2, rmsprop_v2
from keras.models import Sequential
fileName = "C://PythonPrgs/csvFiles/KddTrain_att17.csv"
data = pd.read_csv(fileName)
print(data.shape)
print(data.columns)
# print(data['service'].unique())
lab = LabelEncoder()
catCols = ['protocol_type','flag','service','class', 'AttCat']
# # ohe = OneHotEncoder()
# # data1 = pd.get_dummies(data[catCols])
# # print(data1.head())
data1=pd.DataFrame(data, columns=data.columns)
for i in catCols:
    data1[i] = lab.fit_transform(data1[i])
# data2 = pd.DataFrame(data, columns=data1.columns)
# data2 = pd.concat([data, data1], axis = 1)
y = data1['AttCat']
data2 = data1.drop(['AttCat'], axis=1)
#
#
print("Cols " , data2.columns)
# data2['AttCat'] = lab.fit_transform(data2['AttCat'])
print(data2['AttCat'])
print(data2.shape)
scaler = MinMaxScaler().fit(data2)
X_scaler = scaler.transform(data2)
#
#
X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=42)
# nb = MultinomialNB()
# nbModel = nb.fit(X_train, y_train)
# nbPred = nb.predict(X_test)
# cm = confusion_matrix(y_test, nbPred)
# print("Naive Bayes Confusion Matrix : \n",cm)
# print("Accuracy Score", accuracy_score(y_test, nbPred))
# print("F-score ", f1_score(y_test, nbPred, average='macro'))
# print("Precision : ", precision_score(y_test, nbPred, average='macro'))
# print("Recall: ", recall_score(y_test, nbPred, average='macro'))
# from sklearn.tree  import DecisionTreeClassifier
# tree = DecisionTreeClassifier()
# treeModel = tree.fit(X_train, y_train)
# treePred = tree.predict(X_test)
# cm = confusion_matrix(y_test, treePred)
# print("Decision Tree Confusion Matrix : \n",cm)
# print("Accuracy Score", accuracy_score(y_test, treePred))
# print("F-score ", f1_score(y_test, treePred, average='macro'))
# print("Precision : ", precision_score(y_test, treePred, average='macro'))
# print("Recall: ", recall_score(y_test, treePred, average='macro'))
#
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier()
# rfModel = rf.fit(X_train, y_train)
# rfPred = rf.predict(X_test)
# cm = confusion_matrix(y_test, rfPred)
# print("Random Forest Confusion Matrix : \n",cm)
# print("Accuracy Score", accuracy_score(y_test, rfPred))
# print("F-score ", f1_score(y_test, rfPred, average='macro'))
# print("Precision : ", precision_score(y_test, rfPred, average='macro'))
# print("Recall: ", recall_score(y_test, rfPred, average='macro'))
#
#
model = Sequential()
model.add(Dense(122, activation='relu', input_shape=(X_scaler.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))
# model.add(Dense(FLAGS.nb_classes, activation='softmax'))
# model.add(Dense())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=1)
print("Madhura")
print("Score : ", score)
print("MLP ",model.summary())
#
#
# from sklearn.feature_selection import SelectKBest, chi2
# noofFeat = 25
# bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
# fit = bestfeatures.fit(X_train, y_train)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(data2.columns)
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
# featureScores.columns = ['Specs','Score']   # naming the dataframe columns
# print(featureScores.nlargest(noofFeat,'Score').sort_values(by=['Score'], axis=0,ascending=False))  #print 10 best features
# print("Best Features ")
# print(bestfeatures)
# print("Df scores")
# print(dfscores)
# print("Feature scores")
# print(featureScores)
# feats = featureScores.loc[:15,'Specs']
# print("Features only : \n", feats)
# print("Best Features : \n", bestfeatures)
# selBestFeatures = featureScores.nlargest(noofFeat,'Score')
# print("Selected Best Features : \n", selBestFeatures.iloc[:noofFeat,0])
# selBestColumns = selBestFeatures.iloc[:,0].values
# # print(selColumns)
#
# k=0
# selColumns = []
# # myCols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','num_failed_logins','logged_in','root_shell','su_attempted','num_root','num_file_creations','num_access_files','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','same_srv_rate','diff_srv_rate','role','class']
# for i in selBestFeatures.iloc[:, 0]:
#     selColumns.append(i)
#
# newDf = pd.DataFrame()
# for i in data2.columns:
#     if i in selColumns:
#         newDf[[i]] = data2[[i]]
#
# print(newDf.head())
# print(newDf.columns)
# dataNew = pd.DataFrame(newDf, columns=newDf.columns)
# dataNew['AttCat'] = data1['AttCat']
# print(dataNew.head())
# print(dataNew.columns)
# X = dataNew.drop('AttCat', axis=1)
# y = dataNew['AttCat']
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# nb = MultinomialNB()
# nbModel = nb.fit(X_train, y_train)
# nbPred = nb.predict(X_test)
# cm = confusion_matrix(y_test, nbPred)
# print("With reduced features Naive Bayes Confusion Matrix : \n",cm)
# print("Accuracy Score", accuracy_score(y_test, nbPred))
# print("F-score ", f1_score(y_test, nbPred, average='macro'))
# print("Precision : ", precision_score(y_test, nbPred, average='macro'))
# print("Recall: ", recall_score(y_test, nbPred, average='macro'))
