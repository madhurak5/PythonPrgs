#  To classify images in to different digits
# ------------------------------------
# from keras.datasets import mnist
# (train_imgs, train_labs), (test_imgs, test_labs) = mnist.load_data()
# print(train_imgs.shape)
# print(test_imgs.shape)
# from keras import layers, models
# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(784, )))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# train_imgs = train_imgs.reshape((60000, 28*28))
# train_imgs = train_imgs.astype('float32')/255
# test_imgs = test_imgs.reshape((10000, 28*28))
# test_imgs = test_imgs.astype('float32')/255
# from keras.utils import to_categorical
# train_labs = to_categorical(train_labs)
# test_labs = to_categorical(test_labs)
# network.fit(train_imgs,train_labs,epochs=5, batch_size=128)
# network.summary()
# test_loss, test_acc = network.evaluate(test_imgs, test_labs)
# print("Test Loss : ", test_loss)
# print("Test Accuracy : ", test_acc)
# digit = train_imgs[5]
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
# ---------------------------------------------------------------------------------------------------------------------
from keras.datasets import boston_housing
(train_data, train_labs), (test_data, test_labs) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
mean = train_data.mean(axis=0)
train_data -= mean
stddev = train_data.std(axis=0)
train_data /= stddev

test_data -= mean
test_data /= stddev
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
    input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model


build_model().summary()