import random
from warnings import simplefilter
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt
import cleverhans.tf2.attacks.fast_gradient_method
import cleverhans.tf2.attacks.carlini_wagner_l2
import pandas as pd
import torch.utils.data
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

simplefilter(action='ignore', category=FutureWarning)

fileName = "C://PythonPrgs/csvFiles/KDDTrain.csv"
data = pd.read_csv(fileName)
print(data.shape)
print(data.columns)
print(data.dtypes)
print(data['protocol_type'].unique())
print(data['service'].unique())
print(data['flag'].unique())
catCols = ['protocol_type', 'service', 'flag', 'class']
labEnc = LabelEncoder()
data1 = data.copy()
for i in catCols:
    data1[i] = labEnc.fit_transform(data1[i])
print(data1.dtypes)
print(data1.columns)
print(data1.head())

data1_x = data1.drop('class', axis=1)
data1_y = data1.loc[:, ['class']]
X_train, X_test, y_train, y_test = train_test_split(data1_x, data1_y, test_size=0.2, random_state=42)
train_scale = MinMaxScaler().fit(X_train)
X_train_scaler = train_scale.transform(X_train)
test_scale = MinMaxScaler().fit(X_test)
X_test_scaler = train_scale.transform(X_test)
#
print(X_train_scaler)
print(X_test_scaler)

from sklearn.naive_bayes import BernoulliNB
gnb = BernoulliNB()
gnbModel = gnb.fit(X_train, y_train)
gnbPred = gnb.predict(X_test)
print("Naive Bayes : ",accuracy_score(y_test, gnbPred))
cm = confusion_matrix(y_test, gnbPred)
print(cm)
print("Accuracy Score", accuracy_score(y_test, gnbPred))
print("F-score ", f1_score(y_test, gnbPred))
print("Precision : ", precision_score(y_test, gnbPred))
print("Recall: ", recall_score(y_test, gnbPred))
adv_examples = []
print(X_train.dtypes)
def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost

def gradient_descent(x, y, iterations=1000, learning_rate=0.0001,
                     stopping_threshold=1e-6):
    # Initializing weight, bias, learning rate and iterations
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    # Estimation of optimal parameters
    for i in range(iterations):

        # Making predictions
        y_predicted = (current_weight * x) + current_bias

        # Calculationg the current cost
        current_cost = mean_squared_error(y, y_predicted)

        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost

        costs.append(current_cost)
        weights.append(current_weight)

        # Calculating the gradients
        weight_derivative = -(2 / n) * sum(x * (y - y_predicted))
        bias_derivative = -(2 / n) * sum(y - y_predicted)

        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i + 1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")

    # Visualizing the weights and cost at for all iterations
    plt.figure(figsize=(8, 6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()

    return current_weight, current_bias
#
# estimated_weight, estimated_bias = gradient_descent(X_train, y_train, iterations=2000)
# import tensorflow as tf
# signed_data = tf.sign(estimated_weight)
# print("Signed data : ", signed_data)
# print("np.inf = ", np.inf)
# mult = 0.15 * signed_data
# print("Mult : ", mult)
# pert = X_train + mult
# print("Perturb : ", pert)
# print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

# Making predictions using estimated parameters
# Y_pred = estimated_weight * X + estimated_bias
#
# # Plotting the regression line
# plt.figure(figsize=(8, 6))
# plt.scatter(X, Y, marker='o', color='red')
# plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue', markerfacecolor='red',
#          markersize=10, linestyle='dashed')
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
# # def fgsm_attack(d1, epsilon, data_grad):
#     # sign_data_grad = tf.sign(d1)
#     perturbed_ex = d1 + epsilon * data_grad
#     return perturbed_ex

# train_loader = torch.utils.data.DataLoader(data1_x, batch_size=64, shuffle=True)
# train_features, train_labels = next(iter(train_loader))
# print("Feature batch shape: ", train_features.size())
# print("Labels batch shape: ", train_labels.size())
# d1 = data1_x
# epsilon = 0.15
# data_grad = 0.1
# perturbed_data = fgsm_attack(data1_x, epsilon, data_grad)
# print(perturbed_data)