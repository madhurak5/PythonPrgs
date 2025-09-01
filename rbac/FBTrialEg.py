# import tensorflow.compat.v1 as tf
# import foolbox, torch
# import numpy as np
# import torchvision.models as models
# from keras.applications.vgg19 import VGG19
#
# model = VGG19(weights='imagenet')
# tf.disable_v2_behavior()
#
# # images = tf.placeholder(tf.float32, (None, 224, 224, 3))
# # logits = VGG19(preprocessed)
# # from foolbox.models import TensorFlowModel
# # model = TensorFlowModel(images, logits, bounds = (0, 255))
# # from foolbox.criteria import TargetClassProbability
# #
# # target_class = 22
# # criterion = TargetClassProbability(target_class, p = 0.99)
# # from foolbox.attacks import LBFGSAttack
# # attack = LBFGSAttack(model, criterion)
#
# batch = 10
# network = models.inception_v3(pretrained=True)
# dataset = 'imagenet'
# channels = 3
# size = 224
# classes = 1000
# network.eval()
# fnetwork = foolbox.models.PyTorchModel(network, bounds = (0, 1)) # num_classes = classes,,  channel_axis = 1
# # images, labels = foolbox.utils.samples(fmodel=dataset = dataset, batchsize=batch, data_format='channels_first', bounds=(0,1))
# images, labels = foolbox.utils.samples(fmodel=fnetwork, dataset=dataset, batchsize=batch, data_format='channels_first', bounds=(0,1))
#
# images = images.reshape(batch, channels, size, size)
# print(images.shape)
# print("Labels: ", labels)
# predictions = fnetwork.forward(images).argmax(axis=-1)
# # predictions = fnetwork.f
# print("Predictions: ", predictions)
# print("Accuracy: ", np.mean(predictions == labels))
# already_correct = np.sum(predictions != labels)
# attack = foolbox.attacks.L2DeepFoolAttack(fnetwork, distance=foolbox.distances.Linfinity)
# print(attack)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import tensorflow as tf
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# mpl.rcParams['figure.figsize'] = (8, 8)
# mpl.rcParams['axes.grid'] = False
# pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
# pretrained_model.trainable = False
#
# # ImageNet labels
# decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
# # Helper function to preprocess the image so that it can be inputted in MobileNetV2
# def preprocess(image):
#   image = tf.cast(image, tf.float32)
#   image = tf.image.resize(image, (224, 224))
#   image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
#   image = image[None, ...]
#   return image
#
# # Helper function to extract labels from probability vector
# def get_imagenet_label(probs):
#   return decode_predictions(probs, top=1)[0][0]
# image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
# image_raw = tf.io.read_file(image_path)
# image = tf.image.decode_image(image_raw)
#
# image = preprocess(image)
# image_probs = pretrained_model.predict(image)
# plt.figure()
# plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
# _, image_class, class_confidence = get_imagenet_label(image_probs)
# plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
# plt.show()
# loss_object = tf.keras.losses.CategoricalCrossentropy()
#
# def create_adversarial_pattern(input_image, input_label):
#   with tf.GradientTape() as tape:
#     tape.watch(input_image)
#     prediction = pretrained_model(input_image)
#     loss = loss_object(input_label, prediction)
#
#   # Get the gradients of the loss w.r.t to the input image.
#   gradient = tape.gradient(loss, input_image)
#   # Get the sign of the gradients to create the perturbation
#   signed_grad = tf.sign(gradient)
#   return signed_grad
#
# # Get the input label of the image.
# labrador_retriever_index = 208
# label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
# label = tf.reshape(label, (1, image_probs.shape[-1]))
#
# perturbations = create_adversarial_pattern(image, label)
# plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
#
# def display_images(image, description):
#   _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
#   plt.figure()
#   plt.imshow(image[0]*0.5+0.5)
#   plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
#                                                    label, confidence*100))
#   plt.show()
#
# epsilons = [0, 0.01, 0.1, 0.15]
# descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
#                 for eps in epsilons]
#
# for i, eps in enumerate(epsilons):
#   adv_x = image + eps*perturbations
#   adv_x = tf.clip_by_value(adv_x, -1, 1)
#   display_images(adv_x, descriptions[i])
import torch
from tensorflow.python.ops.parallel_for.gradients import batch_jacobian

  # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
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
# model = Sequential()
# model.add(Dense(64, activation='relu', input_shape=(X_scaler.shape[1],)))
# model.add(Dropout(0.4))
# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(1, activation='softmax'))
# # model.add(Dense(FLAGS.nb_classes, activation='softmax'))
# # model.add(Dense())
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print("MLP ",model.summary())


X_train,X_test, y_train, y_test = train_test_split(X_scaler, y,test_size=0.20, random_state=41)

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
gnbModel = gnb.fit(X_train, y_train)
gnbPred = gnb.predict(X_test)
print("Naive Bayes : ------> ")
cm = confusion_matrix(y_test, gnbPred)
print("Confusion Matrix : \n",cm)
# print("Accuracy Score", accuracy_score(y_test, gnbPred))
# print("F-score ", f1_score(y_test, gnbPred))
# print("Precision : ", precision_score(y_test, gnbPred))
# print("Recall: ", recall_score(y_test, gnbPred))
# mpl.rcParams['figure.figsize'] = (8, 8)
# mpl.rcParams['axes.grid'] = False
pretrained_model = gnbModel
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = gnbPred


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image


# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]


image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg',
                                   'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence * 100))
plt.show()
loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
      tape.watch(input_image)
      prediction = pretrained_model(input_image)
      loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_image)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad


# Get the input label of the image.
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]


def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0] * 0.5 + 0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence * 100))
  plt.show()


epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
              for eps in epsilons]

for i, eps in enumerate(epsilons):
  adv_x = image + eps * perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])