import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import  keras
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from pprint import pprint
data_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
colNames = ["MPG", "Cylinders", "Displacement","Horespower", "Height","Acceleration","Model Year", "Origin"]
raw_dataset = pd.read_csv(data_path, names=colNames, na_values="?", comment="\t",sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())
print(dataset.isna().sum())
dataset = dataset.dropna()
origin = dataset.pop("Origin")
dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0
# print(dataset.tail())
train_dataset = dataset.sample(frac = 0.8, random_state = 0)
test_dataset = dataset.drop(train_dataset.index)
print(train_dataset.shape)
print(test_dataset.shape)
sns.pairplot(train_dataset[["MPG","Cylinders", "Displacement","Height"]], diag_kind="kde")
# plt.show()
train_stats = train_dataset.describe()
print(train_stats.pop("MPG"))
train_stats = train_stats.transpose()
print(train_stats)
train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")
print(test_labels)

def normalize(x):
    return  (x - train_stats['mean'])/train_stats['std']

norm_train_dataset = normalize(train_dataset)
# print(norm_train_dataset)
norm_test_dataset = normalize(test_dataset)
# print(norm_test_dataset)
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=[len(train_dataset.keys())]),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])
optimiser = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss="mse",
              optimizer=optimiser,
              metrics=["mae", "mse"])
print(model.summary())
# model.fit(train_dataset, train_labels, epochs=10)
# test_loss, test_acc = model.evaluate(test_dataset, test_labels)
# print("TEst loss", test_loss)
# print(("Test Acc", test_acc))
# print(tf.__version__)
# mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(train_images.shape)
# print(test_images.shape)
# class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress","Coat", "Sandal", "Shirt", "Sneaker","Bag", "Ankle boot"]
# print(class_names)
# print(len(train_labels))
# print(len(test_labels))
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()
# train_images = train_images/255.0
# test_images = test_images / 255.0
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
#     # plt.colorbar()
# plt.show()
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     # keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])
# model.compile(optimizer="adam",
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(train_images, train_labels, epochs=10)
# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("TEst loss", test_loss)
# print(("Test Acc", test_acc))
# prediction = model.predict((test_images))
# print(prediction[0])
# print(np.argmax(prediction[2]))
# data = pd.read_csv("D:/PythonPrgs/csvFiles/mnist_train.csv")
# pprint(data.columns)
# print(data.shape)
# df_x = data.iloc[:,1:]
# # print(df_x)
# df_y = data.iloc[:,0]
# # print(df_y)
# X_train, X_test,  y_train, y_test = train_test_split(df_x,df_y, test_size=0.20, random_state=41)
# print(X_train.shape)
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# sc = dt.score(X_test, y_test)
# # print("Score of DT : ", sc)
# # print("Score with Training set : ", dt.score(X_train, y_train))
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# rf = RandomForestClassifier(n_estimators=20)
# rf.fit(X_train, y_train)
# scf = rf.score(X_test, y_test)
# print("Score of RF: ", scf)
# bg = BaggingClassifier  (DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)
# bg.fit(X_train, y_train)
# bgSc = bg.score(X_test, y_test)
# print("Bagging Score : ", bgSc)
#
# adb = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=5, learning_rate=1)
# adb.fit(X_train, y_train)
# adbSc = adb.score(X_test, y_test)
# print("AdaBoost Score : ", adbSc)
# lr = LogisticRegression(solver='lbfgs', multi_class='auto')
# dt1 = DecisionTreeClassifier()
# svm = SVC(kernel="poly", degree=2)
# evc = VotingClassifier(estimators=[("lr", lr), ("dt1", dt), ("svm", svm)], voting="hard")
# evc.fit(X_train.iloc[1:15000], y_train.iloc[1:15000])
# evcSc = evc.score(X_test, y_test)
# print("Voting Score : ", evcSc)
#
#

# from sklearn.datasets import fetch_20newsgroups
# newsgroups_train = fetch_20newsgroups(subset="train")
# from pprint import pprint
# targetNames = list(newsgroups_train.target_names)
# print("Target names : \n", targetNames)
# pprint(targetNames)
# print(newsgroups_train.filenames.shape)
# print(newsgroups_train.target.shape)
# cats = ['alt.atheism', 'sci.crypt']
# newsgroups_train1 = fetch_20newsgroups(subset="train", categories=cats)
# pprint(list(newsgroups_train1.target_names))
# print(newsgroups_train1.filenames.shape)
# print(newsgroups_train1.target.shape)
# from keras.datasets import mnist
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print("Shape of training set : ", train_images.shape)
# print("Datatype of training set : ", train_images.dtype)
# train_images = train_images.reshape((60000, 28 *28))
# train_images = train_images.astype('float32')/255
# print("No. of dimensions of training set : ", train_images.ndim)
# print("Shape of Training Set after reshaping : ", train_images.shape)
# print("No. of Training Labels : ",len(train_labels))
# print("Shape of testing set : ", test_images.shape)
# test_images = test_images.reshape(10000, 28* 28)
# test_images = test_images.astype('float32')/255
# print("No. of Testing Labels : ",len(test_labels))
# print("Test Labels : ", test_labels)
# from keras import models
# from keras import layers
# from keras.utils import to_categorical
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# # network.fit(train_images, train_labels, epochs=5, batch_size=128)
# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print("Test Loss : ", test_loss)
# print("Test Accuracy : ", test_acc)
# my_slice = train_images[128 * 10:128 *(10+1)]
# print("Slice shape : ", my_slice.shape)

# from keras.datasets import reuters
# from keras.datasets import imdb
# np_load_old = np.load
# np.load = lambda *a, **k:np_load_old(*a, allow_pickle=True, **k)
# # (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# # print(train_data[0])
# np.load = np_load_old
# print(max([max(sequence) for sequence in train_data] ))
# # word_index = reuters.get_word_index()
# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])
# # print(word_index)
# # print(reverse_word_index)
# # print(decoded_review)
# def vectorize_seq(seq, dimension=10000):
#     results = np.zeros((len(seq), dimension))
#     for i, seq in enumerate(seq):
#         results[i, seq] = 1.
#     return results
#
#
# x_train = vectorize_seq(train_data)
# x_test = vectorize_seq(test_data)
# print(x_train[0])
# y_train = np.asarray(train_labels).astype('float32')
# y_test = np.asarray(test_labels).astype('float32')
# def build_model():
#     from keras import models,layers
#     model = models.Sequential()
#     # model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
#     # model.add(layers.Dense(16, activation='relu'))
#     # model.add(layers.Dense(1, activation='sigmoid'))
#     model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(layers.MaxPooling2D(2,2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(layers.MaxPooling2D(2, 2))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(10, activation='softmax'))
#     from keras import optimizers, losses, metrics
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
#
#
# model = build_model()
# print("Model Summary ", model.summary())
# model.fit(x_train, y_train, epochs=4, batch_size=512)
# results = model.evaluate(x_test, y_test)
# print("Results : ",results)
# x_val = x_train[:10000]
# partial_x_train = x_train[10000:]
# y_val = y_train[:10000]
# partial_y_train = y_train[10000:]
# history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))
# hist_dict = history.history
# print(hist_dict.keys())
# loss_vals = hist_dict['loss']
# val_loss_vals = hist_dict['val_loss']
# acc = hist_dict['acc']
# epochs = range(1, len(acc) + 1)
# import matplotlib.pyplot as plt
# plt.plot(epochs, loss_vals, 'bo', label = "Training Loss")
# plt.plot(epochs, val_loss_vals, 'b', label = "Validation Loss")
# plt.title("Training and Validation Loss")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# plt.clf()
# acc_vals = hist_dict['acc']
# plt.plot(epochs, acc_vals, 'bo', "Training Accuracy")
# val_acc_vals = hist_dict['val_acc']
# plt.plot(epochs,val_acc_vals, 'b', "Validation Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# predictions = model.predict(x_test)
# print("Predictions Shape : ", predictions[0].shape)
# print("Coefficients : ", np.sum(predictions[0]))
# print("Highest probability : ", np.argmax(predictions[0]))
