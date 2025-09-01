import tensorflow.compat.v1 as tf
import numpy as np
import keras
# import tensorflow as tf

from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import pandas as pd
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import TensorFlowClassifier, KerasClassifier, TensorFlowV2Classifier
from art.utils import load_mnist
tf.config.run_functions_eagerly(True)
# Step 1: Load the MNIST dataset

traindata = pd.read_csv('C://PythonPrgs/csvFiles/Files/newFile24AttCatRole.csv',  low_memory=False)
# testdata = pd.read_csv('C://PythonPrgs/csvFiles/KDDTest.csv',  low_memory=False)
lab = LabelEncoder()
catCols = ['protocol_type','flag','service', 'attType', 'AttCat', 'role']
data1=pd.DataFrame(traindata, columns=traindata.columns)
# data2=pd.DataFrame(testdata, columns=testdata.columns)
for i in catCols:
    data1[i] = lab.fit_transform(traindata[i])
    # data2[i] = lab.fit_transform(testdata[i])
# y = data1['class']
data3 = data1.drop(['class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(data1, y, test_size=0.2, random_state=1 )
scaler = MinMaxScaler().fit(X_train)
trainX = scaler.transform(X_train)

# lr.fit(x.reshape(-1, 1), y)
print(trainX.shape)
# print(y_train)
y1 = y_train
print(y1.shape)
# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

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
#
# # Step 3: Create the ART classifier
#
# classifier = TensorFlowClassifier(
#     # clip_values=(min_pixel_value, max_pixel_value),
#     input_ph=input_ph,
#     output=logits,
#     labels_ph=labels_ph,
#     train=train,
#     loss=loss,
#     learning=None,
#     sess=sess,
#     preprocessing_defences=[],
# )
classifier = TensorFlowV2Classifier(
    # clip_values=(min_pixel_value, max_pixel_value),
    model=x,
    nb_classes=2,
    input_shape=[None, 32, 32, 1],
    loss_object=loss,
    train_step=1,
)
print("Classifier : ", classifier)
# # Step 4: Train the ART classifier
# # classifier.fit(X_train, y_train, batch_size=64, nb_epochs=3)
# b = pd.DataFrame(data1['class'])
# classifier.fit(X_train, y_train, batch_size=128, nb_epochs=10)
# print(classifier)

from art.attacks.evasion.saliency_map import SaliencyMapMethod
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
print("Hello ------------------------")
# # Step 5: Evaluate the ART classifier on benign test examples
#
# predictions = classifier.predict(x_test)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on benign test examples: {}%".format(accuracy * 100))
#
#
# # Step 6: Generate adversarial test examples
# attack = FastGradientMethod(estimator=classifier, eps=0.2)
# x_test_adv = attack.generate(x=x_test)
# print(x_test_adv)
#
#
# # Step 7: Evaluate the ART classifier on adversarial test examples
#
# predictions = classifier.predict(x_test_adv)
# accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
# print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
#
# print("Hello done")
