from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# define example
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
df = pd.read_csv("C://PythonPrgs/csvFiles/KDDTrain.csv")
catCols = ['protocol_type', 'flag', 'service']
dataset_catCols = df[catCols]
print(dataset_catCols.head())
lab = LabelEncoder()
df['protocol_type']  = lab.fit_transform(df['protocol_type'])
print(df['protocol_type'] )
df['flag']  = lab.fit_transform(df['flag'])
print(df['flag'] )
df['service']  = lab.fit_transform(df['service'])
print(df['service'] )
df['class']  = lab.fit_transform(df['class'])
print(df['class'] )
print(df.head())
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 )
scaler = MinMaxScaler().fit(X_train)
trainX = scaler.transform(X_train)
# # protocol type
# unique_protocol=sorted(df.protocol_type.unique())
# string1 = 'protocol_type_'
# unique_protocol2=[string1 + x for x in unique_protocol]
# # service
# unique_service=sorted(df.service.unique())
# string2 = 'service_'
# unique_service2=[string2 + x for x in unique_service]
# # flag
# unique_flag=sorted(df.flag.unique())
# string3 = 'flag_'
# unique_flag2=[string3 + x for x in unique_flag]
# # put together
# dumcols=unique_protocol2 + unique_service2 + unique_flag2
# print(dumcols)
# print(len(dumcols))
#
# #Transform categorical features into numbers using LabelEncoder()
# dataset_train_categorical_values_enc=dataset_catCols.apply(LabelEncoder().fit_transform)
# print(dataset_train_categorical_values_enc.head())
# print(dataset_train_categorical_values_enc.shape)
# enc = OneHotEncoder()
# dataset_train_categorical_values_encenc = enc.fit_transform(dataset_train_categorical_values_enc)
# print(dataset_train_categorical_values_encenc.shape)
# dataset_train_cat_data = pd.DataFrame(dataset_train_categorical_values_encenc.toarray(), columns=dumcols)
# # print(dataset_train_cat_data.head())
# newdf=df.join(dataset_train_cat_data)
# print(newdf.shape)
# print(newdf.head())
# print(newdf.columns)
# newdf = newdf.drop(['protocol_type', 'service', 'flag'], axis=1)
# # newdf = newdf.drop([catCols], axis=1)
# print(newdf.shape)
# # # print(newdf.head())
# classdf = newdf['class']
# newclassdf = classdf.replace({'normal':0, 'anomaly':1})
# newdf['class'] = newclassdf
# print("New CLass df: ", newdf['class'].head())
# y = newdf['class']
# colNames1 = list(newdf)
# print("colnames1 : ", colNames1)
# data3 = newdf.drop(['class'], axis=1)
# x = newdf[newdf.columns].values
# dummies = newdf['class'].values
# y = dummies
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1 )
# scaler = MinMaxScaler().fit(X_train)
# trainX = scaler.transform(X_train)
# print("Hello*******************")
# print(trainX.shape)
# # print(y_train)
# y1 = y_train
#
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
# from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
# # naive = GaussianNB(priors=None, var_smoothing=1e-09)# Accuracy : 0.9895614209168486
# naive =BernoulliNB(alpha=1.0,binarize=.0, fit_prior=False)
# naive.fit(X_train, y_train)
# naivePred = naive.predict(X_test)
#
# cm = confusion_matrix(y_test, naivePred)
# print("Confusion Matrix (Naive Bayes) : ", cm)
# print("Accuracy (Naive Bayes) : %.3f"% accuracy_score(y_test, naivePred))
# print("Precision : %.3f"%precision_score(y_test, naivePred))
# print("Recall : %.3f"%recall_score(y_test, naivePred))
# print("F1-score : %.3f"% f1_score(y_test, naivePred))

# from sklearn.tree import DecisionTreeClassifier
# # tree = DecisionTreeClassifier(criterion = "entropy", max_depth=10)
# tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
# tree.fit(X_train, y_train)
# treePred = tree.predict(X_test)
# cmTree = confusion_matrix(y_test, treePred)
# print("Confusion Matrix (Decision Tree): ", cmTree)
# print("Accuracy (Decision Tree) : %.3f"% accuracy_score(y_test, treePred))
# print("Precision : %.3f"% precision_score(y_test, treePred))
# print("Recall : %.3f"%recall_score(y_test, treePred))
# print("F1-score : %.3f"%f1_score(y_test, treePred))
#
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=4, max_features='sqrt')
# rf.fit(X_train, y_train)
# rfPred = rf.predict(X_test)
# rfcm = confusion_matrix(y_test, rfPred)
# print("Confusion Matrix (Random Forest): ", rfcm)
# print("Accuracy (Random Forest)  :%.2f"% (100*accuracy_score(y_test,  rfPred)))
# print("Precision : %.3f"% precision_score(y_test, rfPred))
# print("Recall : %.3f"%recall_score(y_test, rfPred))
# print("F1-score : %.3f"%f1_score(y_test, rfPred))
# RandomForestClassifier(n_estimators=200, bootstrap = True, max_features = 'sqrt')

# from sklearn.linear_model import LogisticRegression
# # logReg = LogisticRegression(solver='lbfgs', max_iter=100)
# logReg = LogisticRegression(solver='lbfgs', max_iter=300)
# logReg.fit(X_train, y_train)
# logPred = logReg.predict(X_test)
# cmLog = confusion_matrix(y_test, logPred)
# print("Confusion Matrix (LogReg) : ", cmLog)
# print("Accuracy (LogReg) :%.2f "% (100*accuracy_score(y_test, logPred)))
# print("Precision : ", precision_score(y_test, logPred))
# print("Recall : ", recall_score(y_test, logPred))
# print("F1-score : ", f1_score(y_test, logPred))
# #
# from sklearn.neural_network import MLPClassifier
# nnModel = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# nnPred = nnModel.predict(X_test)
# cmNn = confusion_matrix(y_test, nnPred)
# print("Confusion Matrix (NN) : ", cmNn)
# print("Accuracy (NN) :", accuracy_score(y_test, nnPred))
# print("Precision : ", precision_score(y_test, nnPred))
# print("Recall : ", recall_score(y_test, nnPred))
# print("F1-score : ", f1_score(y_test, nnPred))
#
# #
newdf = df.copy()
from sklearn.feature_selection import SelectPercentile, f_classif
np.seterr(divide='ignore', invalid='ignore');
colNames = newdf.columns
selector=SelectPercentile(f_classif, percentile=60)
X_new = selector.fit_transform(newdf, y)
X_new.shape
Y_new = newdf['class']
true=selector.get_support()
newcolindex_DoS=[i for i, x in enumerate(true) if x]
newcolname_DoS=list( colNames[i] for i in newcolindex_DoS )
print("Dos features " ,newcolname_DoS)
anotherDf = newdf[newcolname_DoS]
print("Another Df ")
print(anotherDf.shape)
anotherDf['class'] = newdf['class']
y = anotherDf['class']
true=selector.get_support()
newcolindex_Probe=[i for i, x in enumerate(true) if x]
newcolname_Probe=list( colNames[i] for i in newcolindex_Probe )
print("Probe " ,newcolname_Probe)
print(anotherDf.shape)
print(anotherDf.head())

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
#
# # Create a decision tree classifier. By convention, clf means 'classifier'
# clf = DecisionTreeClassifier(random_state=0)
#
# #rank all features, i.e continue the elimination until the last one
# rfe = RFE(clf, n_features_to_select=1)
# rfe.fit(newdf, newdf['class'].astype('int'))
# print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), colNames1)))
#
#
#
from sklearn.feature_selection import RFE
clf = DecisionTreeClassifier(random_state=0)
rfe = RFE(estimator=clf, n_features_to_select=13, step=1)
rfe.fit(newdf, y.astype(int))
X_rfeDoS=rfe.transform(newdf)
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)
print("RFE cols ", rfecolname_DoS)
rfeDf = df[rfecolname_DoS]
print("Rfe Df shape", rfeDf.shape)
print("Rfe ", rfeDf.head())
# # # ['class', 'service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']
X_train, X_test, y_train, y_test = train_test_split(rfeDf, y, test_size=0.2, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(anotherDf, y, test_size=0.2, random_state=1)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
# naive = GaussianNB() # Accuracy : 0.9895614209168486
naive =BernoulliNB(alpha=1.0,binarize=.0, fit_prior=False)
naive.fit(X_train, y_train)
naivePred = naive.predict(X_test)
#
cm = confusion_matrix(y_test, naivePred)
print("Confusion Matrix (Naive Bayes - reduced) : ", cm)
print("Accuracy (Naive Bayes) : %.3f"%accuracy_score(y_test, naivePred))
print("Precision : %.3f"% precision_score(y_test, naivePred))
print("Recall : %.3f"%recall_score(y_test, naivePred))
print("F1-score : %.3f"% f1_score(y_test, naivePred))
#
from sklearn.neural_network import MLPClassifier
# # # nnModel = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
nnModel = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=75, activation='relu') # solve = lbfgs, adam, sgd
nnModel.fit(X_train, y_train)
nnPred = nnModel.predict(X_test)
cmNn = confusion_matrix(y_test, nnPred)
print("Confusion Matrix (NN - reduced) : ", cmNn)
print("Accuracy (NN) : %.3f"% accuracy_score(y_test, nnPred))
print("Precision :  %.3f"% precision_score(y_test, nnPred))
print("Recall :  %.3f"%recall_score(y_test, nnPred))
print("F1-score : %.3f"% f1_score(y_test, nnPred))


# from sklearn.model_selection import cross_val_score, RepeatedKFold
# sco = cross_val_score(nnModel, X_train, y_train, cv = 5)
# print("Cross Val Score ", sco)
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# scores = cross_val_score(nnModel, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
# # report performance
# print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
#
#
# #
from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(solver='lbfgs', max_iter=50)
logReg.fit(X_train, y_train)
logPred = logReg.predict(X_test)
cmLog = confusion_matrix(y_test, logPred)
print("Confusion Matrix (LogReg - reduced) : ", cmLog)
print("Accuracy (LogReg) :", accuracy_score(y_test, logPred))
print("Precision : ", precision_score(y_test, logPred))
print("Recall : ", recall_score(y_test, logPred))
print("F1-score : ", f1_score(y_test, logPred))


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=2, max_features='sqrt')
rf.fit(X_train, y_train)
rfPred = rf.predict(X_test)
rfcm = confusion_matrix(y_test, rfPred)
print("Confusion Matrix (Random Forest): ", rfcm)
print("Accuracy (Random Forest) Reduced  :%.2f"% (100*accuracy_score(y_test,  rfPred)))
print("Precision : %.3f"% precision_score(y_test, rfPred))
print("Recall : %.3f"%recall_score(y_test, rfPred))
print("F1-score : %.3f"%f1_score(y_test, rfPred))
# RandomForestClassifier(n_estimators=200, bootstrap = True, max_features = 'sqrt')


from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier(criterion = "entropy", max_depth=10)
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
tree.fit(X_train, y_train)
treePred = tree.predict(X_test)
cmTree = confusion_matrix(y_test, treePred)
print("Confusion Matrix (Decision Tree) _ Reduced: ", cmTree)
print("Accuracy (Decision Tree) : %.3f"% accuracy_score(y_test, treePred))
print("Precision : %.3f"% precision_score(y_test, treePred))
print("Recall : %.3f"%recall_score(y_test, treePred))
print("F1-score : %.3f"%f1_score(y_test, treePred))

from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers
x_train, x_test, y_train, y_test = train_test_split(anotherDf, y, test_size=0.2, random_state=42)

# Create neural net
model = Sequential()

model.add(layers.Dense(20))
model.add(layers.Dense(50))
model.add(layers.Dense(10))

model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(1,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
#                         patience=5, verbose=1, mode='auto',
#                            restore_best_weights=True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),verbose=2,epochs=10) # callbacks=[monitor],
model.fit(x_train, y_train, epochs = 10,  verbose=2)
print(model.summary)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow as tf
from art.estimators.classification import KerasClassifier, TensorFlowClassifier
classifier = KerasClassifier(model=model, use_logits=False)
print(classifier)

# Train ART classifier
from art.attacks.evasion import FastGradientMethod
preds = classifier.predict(x_test) # Eval ART Classifier
print(preds)
print("hi")
accu = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis = 1))/len(y_test)
print(accu)

print("Accuracy on benign test examples : {}%".format(accu*100) )

attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x = x_test)

predictions = classifier.predict(x_test_adv)
accuracy1 = np.sum(np.argmax(predictions) == np.argmax(y_test))/len(y_test)
print("Accuracy on adversarial test examples : {}%".format(accuracy1*100) )
