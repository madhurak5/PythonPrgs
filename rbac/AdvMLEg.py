import tensorflow as tf
import neural_structured_learning as nsl
import cleverhans
from cleverhans import utils
# Prepare data.t
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
import pycaret
from pycaret import anomaly

#Createa base model--sequential, functional, orsubclass.
model=tf.keras.Sequential([
    tf.keras.Input((28,28),name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    ])
#
# # Wrap the model with adversarial regularization.
adv_config=nsl.configs.make_adv_reg_config(multiplier=0.2,adv_step_size=0.05)
adv_model=nsl.keras.AdversarialRegularization(model,adv_config=adv_config)
#
# # Compile, train, and evaluate.
adv_model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
adv_model.fit({'feature':x_train,'label':y_train},batch_size=32,epochs=10)
adv_model.evaluate({'feature':x_test,'label':y_test})
#  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
from pycaret.anomaly import *
from sklearn.datasets import load_breast_cancer
df = load_breast_cancer(as_frame=True)['data']
df_train = df.iloc[:-10]
df_unseen = df.tail(10)
anom = setup(data=df_train, silent=True)
print(models())
anom_model = create_model(model='iforest', fraction=0.05)
results = assign_model(anom_model)
print(results)
plot_model(anom_model, plot='umap')
#  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
pretrained_model.trainable = False
#
# # ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
# # Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...]
  return image
#
# # Helper function to extract labels from probability vector
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()
loss_object = tf.keras.losses.CategoricalCrossentropy()
#
def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)
#
#   # Get the gradients of the loss w.r.t to the input image.
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
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()
  epsilons = [0, 0.01, 0.1, 0.15]
  descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                  for eps in epsilons]

  for i, eps in enumerate(epsilons):
      adv_x = image + eps * perturbations
      adv_x = tf.clip_by_value(adv_x, -1, 1)
      display_images(adv_x, descriptions[i])

#  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# from pathlib import Path
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import MNIST
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def mean_squared_error(y_true, y_predicted):
#     # Calculating the loss or cost
#     cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
#     return cost
#
#
# # Gradient Descent Function
# # Here iterations, learning_rate, stopping_threshold
# # are hyperparameters that can be tuned
# def gradient_descent(x, y, iterations=1000, learning_rate=0.0001,
#                      stopping_threshold=1e-6):
#     # Initializing weight, bias, learning rate and iterations
#     current_weight = 0.1
#     current_bias = 0.01
#     iterations = iterations
#     learning_rate = learning_rate
#     n = float(len(x))
#
#     costs = []
#     weights = []
#     previous_cost = None
#
#     # Estimation of optimal parameters
#     for i in range(iterations):
#
#         # Making predictions
#         y_predicted = (current_weight * x) + current_bias
#
#         # Calculationg the current cost
#         current_cost = mean_squared_error(y, y_predicted)
#
#         # If the change in cost is less than or equal to
#         # stopping_threshold we stop the gradient descent
#         if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
#             break
#
#         previous_cost = current_cost
#
#         costs.append(current_cost)
#         weights.append(current_weight)
#
#         # Calculating the gradients
#         weight_derivative = -(2 / n) * sum(x * (y - y_predicted))
#         bias_derivative = -(2 / n) * sum(y - y_predicted)
#
#         # Updating weights and bias
#         current_weight = current_weight - (learning_rate * weight_derivative)
#         current_bias = current_bias - (learning_rate * bias_derivative)
#
#         # Printing the parameters for each 1000th iteration
#         print(f"Iteration {i + 1}: Cost {current_cost}, Weight \
#         {current_weight}, Bias {current_bias}")
#
#     # Visualizing the weights and cost at for all iterations
#     plt.figure(figsize=(8, 6))
#     plt.plot(weights, costs)
#     plt.scatter(weights, costs, marker='o', color='red')
#     plt.title("Cost vs Weights")
#     plt.ylabel("Cost")
#     plt.xlabel("Weight")
#     plt.show()
#
#     return current_weight, current_bias
#
#
# def main():
#     # Data
#     X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
#                   55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
#                   45.41973014, 54.35163488, 44.1640495, 58.16847072, 56.72720806,
#                   48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
#     Y = np.array([31.70700585, 68.77759598, 62.5623823, 71.54663223, 87.23092513,
#                   78.21151827, 79.64197305, 59.17148932, 75.3312423, 71.30087989,
#                   55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
#                   60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])
#
#     # Estimating weight and bias using gradient descent
#     estimated_weight, eatimated_bias = gradient_descent(X, Y, iterations=2000)
#     import tensorflow as tf
#     signed_data = tf.sign(estimated_weight)
#     print("Signed data : ", signed_data)
#     print("np.inf = ", np.inf)
#     mult = 0.15 * signed_data
#     print("Mult : ", mult)
#     pert = X + mult
#     print("Perturb : ", pert)
#     print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {eatimated_bias}")
#
#     # Making predictions using estimated parameters
#     Y_pred = estimated_weight * X + eatimated_bias
#
#     # Plotting the regression line
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X, Y, marker='o', color='red')
#     plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue', markerfacecolor='red',
#              markersize=10, linestyle='dashed')
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()





#  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------