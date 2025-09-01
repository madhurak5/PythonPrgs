import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scikitplot as skplt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df = pd.read_csv("C://PythonPrgs/csvFiles/KddTrain_att17.csv")
# # def process_data():
# # colNames = ['id', 'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
# #    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
# #    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
# #    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
# #    'num_access_files', 'num_outbound_cmds', 'is_host_login',
# #    'is_guest_login', 'count', 'srv_count', 'serror_rate',
# #    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
# #    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
# #    'dst_host_srv_count', 'dst_host_same_srv_rate',
# #    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
# #    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
# #    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
# #    'dst_host_srv_rerror_rate', 'class', 'no', 'AttCat']
# # df = pd.read_csv("C://PythonPrgs/csvFiles/KddTrain_att17.csv", header=0, names=colNames)
# # col_norm = ['id', 'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
# #    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
# #    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
# #    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
# #    'num_access_files', 'num_outbound_cmds', 'is_host_login',
# #    'is_guest_login', 'count', 'srv_count', 'serror_rate',
# #    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
# #    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
# #    'dst_host_srv_count', 'dst_host_same_srv_rate',
# #    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
# #    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
# #    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
# #    'dst_host_srv_rerror_rate', 'class', 'no','AttCat']
lab = LabelEncoder()
catCols = ['protocol_type', 'flag', 'service', 'class', 'AttCat']
# ohe = OneHotEncoder()
# data1 = pd.get_dummies(data[catCols])
# print(data1.head())
data1 = pd.DataFrame(df, columns=df.columns)
for i in catCols:
    data1[i] = lab.fit_transform(data1[i])
# data2 = pd.DataFrame(data, columns=data1.columns)
# data2 = pd.concat([data, data1], axis = 1)
y = data1['AttCat']
data2 = data1.drop(['AttCat'], axis=1)
print(data2.head())
# import  keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# model = Sequential()
#
# df1_norm = StandardScaler().fit(data2)
# X_scaler = df1_norm.transform(data2)
# # y = df1_norm['AttCat']
# X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=42)
# #     # return X_train, X_test, y_train, y_test
# model.add(Dense(122, activation='relu', input_shape=(X_train.shape[1],)))
#
# model.add(Dropout(0.4))
# model.add(Dense(122, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(1, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print("MLP ",model.summary())
# hist = model.fit(X_train, y_train,batch_size=5, epochs=2,verbose=1, shuffle=True)
# print(hist)
# # X_train, X_test, y_train, y_test = process_data()
# feature_col = df.columns
# # input_func = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size = 50, num_epochs=1000, shuffle= True)
# input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=50, num_epochs=1000, shuffle=True)
# # eval_func = tf.compat.v1.estimator.inputs.pandas_input_fn(X_test, y_test, batch_size =50, num_epochs = 1, shuffle = False)
# # predict_func =  tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test, num_epochs =1 , shuffle=False)
# # dnnmodel = tf.estimator.DNNClassifier(hidden_units=[20, 20], feature_col=df.columns, n_classes=2, activation_fn=tf.nn.softmax, dropout=None, optimizer=tf.train.AdamOptimizer(learning_rate=0.01))
# # history = dnnmodel.train(input_fn=input_func, steps=500)
# dnnmodel = tf.compat.v1.estimator.DNNClassifier(hidden_units=[20,20], feature_columns=df.columns, n_classes=2, activation_fn=tf.nn.softmax, dropout=None, optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01))
#
# print(dnnmodel.evaluate(input_fn=input_func))

#
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
# from tensorflow import keras
# import pandas as pd
# from  keras.models import Sequential
# from keras.layers import Dense, Dropout
# from matplotlib import pyplot as plt
# df = pd.read_csv("C://PythonPrgs/csvFiles/KddTrain_att17.csv")
# df.head()
#
# lab = LabelEncoder()
# catCols = ['protocol_type', 'flag', 'service', 'class', 'AttCat']
# # ohe = OneHotEncoder()
# # data1 = pd.get_dummies(data[catCols])
# # print(data1.head())
# data1 = pd.DataFrame(df, columns=df.columns)
# for i in catCols:
#     data1[i] = lab.fit_transform(data1[i])
# # data2 = pd.DataFrame(data, columns=data1.columns)
# # data2 = pd.concat([data, data1], axis = 1)
# y = data1['AttCat']
# data2 = data1.drop(['AttCat'], axis=1)
# print(data2.head())
# df1_norm = StandardScaler().fit(data2)
# X_scaler = df1_norm.transform(data2)
# # y = df1_norm['AttCat']
# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import TensorFlowClassifier
# from art.utils import load_mnist
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X_scaler, y, test_size=0.2, random_state=42)
# model = Sequential()
# model.add(Dense(1, input_shape=(X_train.shape[1],), activation='relu', kernel_initializer='ones', bias_initializer='zeros'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs =10)
# model.evaluate(X_test, y_test)
# model.predict(X_test)
# print(y_test)
# coeff, intercept = model.get_weights()
# print(coeff, intercept)
#
# def sigmoid(x):
#     import math
#     return 1/(1+math.exp(-x))
#
# print(sigmoid(18))


# import tensorflow.compat.v1 as tf
# tf.compat.v1.estimator.inputs.p
# import  tensorflow as tf
#
# import numpy as np
#
# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import TensorFlowClassifier
# from art.utils import load_mnist
#
# # Step 1: Load the MNIST dataset
#
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
#
# # Step 2: Create the model
#
# tf.compat.v1.disable_eager_execution()
# # input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# input_ph = tf.compat.v1.placeholder(tf.float32,shape=None) #, shape=[None, 28,28,1])
# # labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
# labels_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, 10])
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# x = Conv2D(input_shape=input_ph,filters=4, kernel_size=5, activation=tf.nn.relu)
# x = MaxPooling2D(x, 2,2 )
# x = Conv2D(x, filters=10, kernel_size=5, activation=tf.nn.relu)
# x = MaxPooling2D(x, 2,2)
# x =  Flatten(x)
# x = Dense(x, 100, activation=tf.nn.relu)
# logits = Dense(x, 10)
#
# loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
# loss
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # Step 3: Create the ART classifier
#
# classifier = TensorFlowClassifier(
#     clip_values=(min_pixel_value, max_pixel_value),
#     input_ph=input_ph,
#     output=logits,
#     labels_ph=labels_ph,
#     train=train,
#     loss=loss,
#     learning=None,
#     sess=sess,
#     preprocessing_defences=[],
# )
#
# # Step 4: Train the ART classifier
#
# classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
#
# # Step 5: Evaluate the ART classifier on benign test examples
#
# predictions = classifier.predict(x_test)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on benign test examples: {}%".format(accuracy * 100))
#
# # Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
# x_test_adv = attack.generate(x=x_test)
#
# # Step 7: Evaluate the ART classifier on adversarial test examples
#
# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


# import tensorflow as tf
#
# tf.compat.v1.disable_eager_execution()
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
# from tensorflow.keras.losses import categorical_crossentropy
# from tensorflow.keras.optimizers import Adam
# import numpy as np
#
# from art.attacks.evasion import FastGradientMethod
# from art.estimators.classification import KerasClassifier
# from art.utils import load_mnist
#
# # Step 1: Load the MNIST dataset
#
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
#
# # Step 2: Create the model
#
# model = Sequential()
# model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(100, activation="relu"))
# model.add(Dense(10, activation="softmax"))
#
# model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])
#
# # Step 3: Create the ART classifier
#
# classifier = KerasClassifier(model=model, clip_values=(min_pixel_value, max_pixel_value), use_logits=False)
#
# # Step 4: Train the ART classifier
#
# classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)
#
# # Step 5: Evaluate the ART classifier on benign test examples
#
# predictions = classifier.predict(x_test)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on benign test examples: {}%".format(accuracy * 100))
#
# # Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
# x_test_adv = attack.generate(x=x_test)
#
# # Step 7: Evaluate the ART classifier on adversarial test examples
#
# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding
from keras.layers import Convolution1D,MaxPooling1D, Flatten
from keras.datasets import imdb
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.utils import np_utils
import numpy as np
import h5py
from keras import callbacks
from keras.layers import LSTM, GRU, SimpleRNN
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error
                             ,mean_absolute_error)
from sklearn import metrics

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

traindata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTrain.csv')
testdata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTest.csv')

# df.head()
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
lab = LabelEncoder()
catCols = ['protocol_type', 'flag', 'service', 'class']
# ohe = OneHotEncoder()
# data1 = pd.get_dummies(data[catCols])
# print(data1.head())
data1 = pd.DataFrame(traindata, columns=traindata.columns)
for i in catCols:
    data1[i] = lab.fit_transform(data1[i])
# data2 = pd.DataFrame(data, columns=data1.columns)
# data2 = pd.concat([data, data1], axis = 1)
y = data1['class']
data2 = data1.drop(['class'], axis=1)
print(data2.head())
df1_norm = StandardScaler().fit(data2)
X_scaler = df1_norm.transform(data2)
# y = df1_norm['AttCat']
X = traindata.iloc[:,1:42]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

# scaler = Normalizer().fit(X)
# trainX = scaler.transform(X)
#
# scaler = Normalizer().fit(T)
# testT = scaler.transform(T)
#
# y_train = np.array(Y)
# y_test = np.array(C)
X_train = np.reshape(X_scaler, (X_scaler.shape[0],X_scaler.shape[1],1))
# X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))

lstm_output_size = 70

cnn = Sequential()
cnn.add(Convolution1D(64, 3,activation="relu",input_shape=(41, 1)))
cnn.add(MaxPooling1D (pool_size=(3)))
cnn.add(LSTM(lstm_output_size))
cnn.add(Dropout(0.1))
cnn.add(Dense(1, activation="sigmoid"))
cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])

# train
# checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn1results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
# csv_logger = CSVLogger('results/cnn1results/cnntrainanalysis1.csv',separator=',', append=False)
# cnn.fit(X, Y, epochs=10,validation_data=(C, T),callbacks=[checkpointer,csv_logger])
cnn.fit(X, Y,batch_size=10, epochs=5)
cnn.save("results/cnn1results/cnn_model.hdf5")
cnn.load_weights("results/cnn1results/checkpoint-947.hdf5")
cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
loss, accuracy = cnn.evaluate(X_test, y_test)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
