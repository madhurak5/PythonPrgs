import numpy as np

from warnings import simplefilter
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from art.estimators.classification import TensorFlowClassifier, KerasClassifier, TensorFlowV2Classifier
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder
import pandas as pd
simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error
                             ,mean_absolute_error)
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
use_cuda = True
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
traindata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTrain.csv',  low_memory=False)
testdata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTest.csv',  low_memory=False)
lab = LabelEncoder()
ohe = OneHotEncoder()

print(traindata.dtypes)
catCols = ['protocol_type','flag','service', 'class'] #'attType','AttCat', 'role',
data1=pd.DataFrame(traindata, columns=traindata.columns)
# data2=pd.DataFrame(testdata, columns=testdata.columns)
for i in catCols:
    data1[i] = lab.fit_transform(traindata[i])
    print(data1[i])
    # data2[i] = lab.fit_transform(testdata[i])
y = data1['class']
data3 = data1.drop(['class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data1, y, test_size=0.2, random_state=1 )
scaler = MinMaxScaler().fit(X_train)
trainX = scaler.transform(X_train)
print("Hello*******************")
# print(trainX.shape)
# # print(y_train)
y1 = y_train
print(y_test[y_test.index][0])
#
# scaler = MinMaxScaler().fit(data2)
# testT = scaler.transform(data2)
# print(trainX.shape)
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
naive = BernoulliNB() # Accuracy : 0.9895614209168486
naive.fit(X_train, y_train)
naivePred = naive.predict(X_test)

cm = confusion_matrix(y_test, naivePred)
print("Confusion Matrix (Naive Bayes) : ", cm)
print("Accuracy (Naive Bayes):", accuracy_score(y_test, naivePred))
print("Precision : ", precision_score(y_test, naivePred))
print("Recall : ", recall_score(y_test, naivePred))
print("F1-score : ", f1_score(y_test, naivePred))
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4)
tree.fit(X_train, y_train)
treePred = tree.predict(X_test)
cmTree = confusion_matrix(y_test, treePred)
print("Confusion Matrix (Decision Tree): ", cmTree)
print("Accuracy (Decision Tree):", accuracy_score(y_test, treePred))
print("Precision : ", precision_score(y_test, treePred))
print("Recall : ", recall_score(y_test, treePred))
print("F1-score : ", f1_score(y_test, treePred))
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(solver='lbfgs', max_iter=100)
logReg.fit(X_train, y_train)
logPred = logReg.predict(X_test)
cmLog = confusion_matrix(y_test, logPred)
print("Confusion Matrix (LogReg) : ", cmLog)
print("Accuracy (LogReg) :", accuracy_score(y_test, logPred))
print("Precision : ", precision_score(y_test, logPred))
print("Recall : ", recall_score(y_test, logPred))
print("F1-score : ", f1_score(y_test, logPred))
from sklearn.neural_network import MLPClassifier
nnModel = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
nnPred = nnModel.predict(X_test)
cmNn = confusion_matrix(y_test, nnPred)
print("Confusion Matrix (NN) : ", cmNn)
print("Accuracy (NN) :", accuracy_score(y_test, nnPred))
print("Precision : ", precision_score(y_test, nnPred))
print("Recall : ", recall_score(y_test, nnPred))
print("F1-score : ", f1_score(y_test, nnPred))
# from sklearn.feature_selection import RFE, SelectKBest
# from sklearn.svm import SVR
# # >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
# svmModel = SVR(kernel="linear")
# selector = RFE(tree, n_features_to_select=5, step=1)
# selector = selector.fit(X_train, y_train)
# print("Selector : ", selector.support_)
# print("Selector Ranking ", selector.ranking_)
#
# from sklearn.feature_selection import SelectKBest, chi2
# X_new = SelectKBest(chi2, k=20).fit_transform(X_train, y_train)
# # print("X_new : ", X_new)
#
# from sklearn.neural_network import MLPClassifier
# mlpModel = MLPClassifier(hidden_layer_sizes=(5, ), activation='relu',solver='adam', random_state=1, max_iter=1000,verbose=False).fit(X_train, y_train)
# print(mlpModel.score(X_train, y_train))

tf.compat.v1.disable_eager_execution()
# Step 2: Create the model

# input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 1])
# labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
labels_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, 1])
#
x = tf.layers.conv2d(input_ph, filters=42, kernel_size=5, activation=tf.nn.relu)
x = tf.layers.max_pooling2d(x, 2, 2)
x = tf.layers.conv2d(x, filters=10, kernel_size=5, activation=tf.nn.relu)
x = tf.layers.max_pooling2d(x, 2, 2)
x = tf.layers.flatten(x)
x = tf.layers.dense(x, 100, activation=tf.nn.relu)
logits = tf.layers.dense(x, 1)

#
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("Loss : ", loss)
# from art.estimators.classification import KerasClassifier
# classifier = KerasClassifier(model=mlpModel, use_logits=False)
# print(classifier)
classifier = TensorFlowV2Classifier(
    # clip_values=(min_pixel_value, max_pixel_value),
    model=x,
    nb_classes=2,
    input_shape=[None, 32, 32, 1],
    loss_object=loss,
    train_step=1,
)
# print("Classifier : ", classifier)
# classifier.fit(trainX, y_train, batch_size=64, nb_epochs=3)
# print(classifier)
# X = trainX
# y = y1
# noofFeat = 28
# from sklearn.feature_selection import SelectKBest, chi2, f_classif
# bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
# fit = bestfeatures.fit(X_train,y_train)
# dfscores = pd.DataFrame(fit.scores_)
# dfcolumns = pd.DataFrame(X_train.columns)
# featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
# featureScores.columns = ['Specs','Score']   # naming the dataframe columns
# # bestfeatures, dfscores , featureScores = featSelectionKBest(noofFeat)
# feats = featureScores.loc[:15,'Specs']
# print(feats)
# selBestFeatures = featureScores.nlargest(noofFeat,'Score')
#
# # "{:.2f}".format(a_float)
# print("selBestFeatures .... \n", selBestFeatures)
# selBestColumns = selBestFeatures.iloc[:,0].values
# V = 1
# k=0
# selColumns = []
# for i in selBestFeatures.iloc[:, 0]:
#     selColumns.append(i)
# print(selColumns)
# newDf = pd.DataFrame()
# for i in X_train.columns:
#     if i in selColumns:
#         newDf[[i]] = X_train[[i]]
#
#
# # tmp = -90
# # assert (tmp >= 0), "Colder temp"
#
# print("With selected features : .............")
# # X_scaler = minmaxscaling(newDf)
# minMaxScaler = MinMaxScaler().fit(newDf)
# X_scaler = minMaxScaler.transform(newDf)
# X = X_scaler
#
# X_train,X_test, y_train, y_test = train_test_split(X, y1,test_size=0.20, random_state=41)
# # # print("X_train shape reduced features : ", X_train.shape)
# # # from sklearn.crossvalidation import #KFold, cross_val_score
# #
# # # k_fold = KFold(len(y), n_folds=10, shuffle=True, random_state=0)
# # # clf = <any classifier>
# # # print cross_val_score(clf, X, y, cv=k_fold, n_jobs=1)
# # # naiveBayesAlgo()
# # # # print("Calling SVM function....")
# # # svmAlgo(ker)
# newSelCols = newDf.columns
# dsNewOrig = data1.copy()
# newSelDf = pd.DataFrame()
# for i in dsNewOrig.columns:
#     if i in newSelCols:
#         newSelDf[[i]] = dsNewOrig[[i]]
# # newDf['attType'] = dsNewOrig['attType']
# # newSelDf['AttCat'] = ''
# # newSelDf['attType'] = dsNewOrig['attType']
# # myNewDf = pd.read_csv("C://PythonPrgs/csvFiles/finalFile.csv")
# # print("nyNewDf shape ",  myNewDf.shape,"\n", myNewDf.head())
# lab1 = LabelEncoder()
# X_enc_new = pd.DataFrame(newSelDf)
#
# catCols_new = [ 'flag','service','class']
# for i in catCols_new:
#     X_enc_new[i] = lab.fit_transform(X_enc_new[i])
# y = X_enc_new.loc[:,'class']
# # X_enc_new = X_enc_new.drop(['id', 'class','AttCat'], axis=1) #'attType','class','role'
# minMaxScaler = MinMaxScaler().fit(X_enc_new)
# X_scaler = minMaxScaler.transform(X_enc_new)
# # X_scaler = minmaxscaling(X_enc_new)
# X = X_scaler
# # X = X_enc_new
#
# # print("X from finalFile modified Part 1 : ", X.shape)
# print("_______________________ PART - I ___________________")
# X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# naive = BernoulliNB() # Accuracy : 0.5345901964675531
# naive.fit(X_train, y_train)
# naivePred = naive.predict(X_test)
# # print(X.columns)
# cm = confusion_matrix(y_test, naivePred)
# print("Confusion Matrix (Naive Bayes) : ", cm)
# print("Accuracy (Naive Bayes):", accuracy_score(y_test, naivePred))
# print("Precision : ", precision_score(y_test, naivePred))
# print("Recall : ", recall_score(y_test, naivePred))
# print("F1-score : ", f1_score(y_test, naivePred))
#
# logReg = LogisticRegression(solver='lbfgs', max_iter=100)
# logReg.fit(X_train, y_train)
# logPred = logReg.predict(X_test)
# cmLog = confusion_matrix(y_test, logPred)
# print("Confusion Matrix (LogReg) : ", cmLog)
# print("Accuracy (LogReg) :", accuracy_score(y_test, logPred))
# print("Precision : ", precision_score(y_test, logPred))
# print("Recall : ", recall_score(y_test, logPred))
# print("F1-score : ", f1_score(y_test, logPred))