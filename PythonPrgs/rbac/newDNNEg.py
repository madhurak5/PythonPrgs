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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
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
use_cuda = True
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
traindata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTrain.csv',  low_memory=False)
testdata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTest.csv',  low_memory=False)

lab = LabelEncoder()
catCols = ['protocol_type','flag','service','class']
data1=pd.DataFrame(traindata, columns=traindata.columns)
data2=pd.DataFrame(testdata, columns=testdata.columns)
for i in catCols:
    data1[i] = lab.fit_transform(traindata[i])
    data2[i] = lab.fit_transform(testdata[i])
# # data2 = pd.DataFrame(data, columns=data1.columns)
# # data2 = pd.concat([data, data1], axis = 1)
y = data1['class']
data3 = data1.drop(['class'], axis=1)
X_train1= traindata.iloc[:,1:42]
y_train1 = traindata.iloc[:,0]
X_test = testdata.iloc[:,0]
y_test = testdata.iloc[:,1:42]

scaler = MinMaxScaler().fit(X_train1)
trainX = scaler.transform(X_train1)

scaler = MinMaxScaler().fit(y_test)
testT = scaler.transform(y_test)

y_train = np.array(y_train1)
y_test = np.array(X_test)
X_train = np.reshape(trainX, (trainX.shape[0],trainX.shape[1],1))
X_test = np.reshape(testT, (testT.shape[0],testT.shape[1],1))
#
# lstm_output_size = 70
#
# cnn = Sequential()
# cnn.add(Dense(64, activation='relu',input_shape=(41, 1)))
#
# # cnn.add(Convolution1D(64, 3, activation="relu",input_shape=(41, 1)))
# # cnn.add(MaxPooling1D(pool_length=(2)))
# # cnn.add(LSTM(lstm_output_size))
# cnn.add(Dropout(0.4))
# cnn.add(Dense(64, activation='relu'))
# cnn.add(Dropout(0.4))
# cnn.add(Flatten())
# cnn.add(Dense(1, activation="sigmoid"))
#
# cnn.compile(loss="binary_crossentropy", optimizer="adam",metrics=['accuracy'])
#
# cnn.fit(X_train, y_train, batch_size = 10, epochs = 5)
# y_pred= cnn.predict(X_test)
#
# # train
# checkpointer = callbacks.ModelCheckpoint(filepath="results/cnn1results/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='val_acc',mode='max')
# csv_logger = CSVLogger('/content/results/cnn1results',separator=',', append=False) # results/cnn1results/cnntrainanalysis1.csv
# # cnn.fit(X_train, y_train, nb_epoch=10,callbacks=[checkpointer,csv_logger])
# # cnn.fit(X_train, y_train, epochs=10,validation_data=(X_test, y_test),callbacks=[checkpointer,csv_logger])
# # cnn.compile(optimizer='adam',loss = tf.keras.losses.categorical_crossentropy(y_train, y_pred, from_logits=True))
# # cnn.compile(optimizer='adam',loss = tf.keras.losses.binary_crossentropy(optimizer='rmsprop',metrics=['accuracy']))
# # cnn.save('/content/results/cnn1results')
# cnn.save('/content/results/cnn1results/cnn_model.hdf5')
# # cnn.save("results/cnn1results/cnn_model.hdf5")
# cnn.load_weights("/content/results/cnn1results/cnn_model.hdf5")
#
#
# cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# loss, accuracy = cnn.evaluate(X_test, y_test)
# print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# # y_pred = cnn.predict_classes(X_test)
# # y_pred= cnn.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred , average="micro")
# precision = precision_score(y_test, y_pred , average="micro")
# f1 = f1_score(y_test, y_pred, average="micro")
# # np.savetxt('results/expected1.txt', y_test, fmt='%01d')
# # np.savetxt('results/predicted1.txt', y_pred, fmt='%01d')
#
# print("confusion matrix")
# print("----------------------------------------------")
# print("accuracy")
# print("%.6f" %accuracy)
# print("racall")
# print("%.6f" %recall)
# print("precision")
# print("%.6f" %precision)
# print("f1score")
# print("%.6f" %f1)
# cm = metrics.confusion_matrix(y_test, y_pred)
# print("==============================================")
# print (cm)



# tf.compat.v1.disable_eager_execution()
# # Step 2: Create the model
#
# # input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# input_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])
# # labels_ph = tf.placeholder(tf.int32, shape=[None, 10])
# labels_ph = tf.compat.v1.placeholder(tf.int32, shape=[None, 10])
# x = tf.compat.v1.layers()
# x = tf.layers.conv2d(input_ph, filters=4, kernel_size=5, activation=tf.nn.relu)
# x = tf.layers.max_pooling2d(x, 2, 2)
# x = tf.layers.conv2d(x, filters=10, kernel_size=5, activation=tf.nn.relu)
# x = tf.layers.max_pooling2d(x, 2, 2)
# x = tf.layers.flatten(x)
# x = tf.layers.dense(x, 100, activation=tf.nn.relu)
# logits = tf.layers.dense(x, 10)
#
# loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print("Hello")

# from art.attacks.evasion import FastGradientMethod
# attFGSM = FastGradientMethod(estimator=cnn, eps=0.1, eps_step=0.3, targeted=False, batch_size=32)
# FGSMAttacks = attFGSM.generate(x = X_test)
# print(FGSM)
# # from art.estimators.classification import TensorFlowClassifier, KerasClassifier
# # classifier = KerasClassifier(model=cnn, use_logits=False)  # Create ART classifier
# # classifier.fit(X_train, y_train, batch_size=10, nb_epochs=3) # Train ART classifier
# # print('Classifier : ', classifier)
# preds = cnn.predict(X_test) # Eval ART Classifier
# accu = np.sum(np.argmax(preds, axis=1) == np.argmax(y_test, axis=1))/len(y_test)
# # # Generate adversarial test examples
# print("Accuracy on benign test examples : {}%".format(accu*100) )
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
# x_test_adv = attack.generate(x = X_test)
# # Eval ART Classifier on adversarial test examples
# predictions = classifier.predict(x_test_adv)
# accuracy1 = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))/len(y_test)
# print("Accuracy on adversarial test examples : {}%".format(accuracy1*100) )


import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_mnist

# Step 1: Load the MNIST dataset

# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
X_train, X_test, y_train, y_test = train_test_split(X_train1, y_train1, test_size=0.2, random_state=1)
# Step 2: Create the model

model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=1, activation="relu", input_shape=(23, 23, 4)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=0.01), metrics=["accuracy"])

# Step 3: Create the ART classifier

classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)

# Step 4: Train the ART classifier

classifier.fit(X_train1, y_train1, batch_size=32, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(X_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = FastGradientMethod(estimator=classifier, eps=0.2)
x_test_adv = attack.generate(x=X_test)

# Step 7: Evaluate the ART classifier on adversarial test examples

predictions = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
