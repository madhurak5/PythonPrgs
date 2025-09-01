# Load data, Define Keras model, Compile, Fit, Evaluate, Make Predctions
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# Load dataset
dataset = loadtxt("C:\PythonPrgs\csvFiles\KDDTrain.csv", delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]
# # Define Keras model
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#
# # Compile Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # Fit Keras model
model.fit(X, y, epochs=150, batch_size=10)
#
# # Evaluate Keras model
_, acc = model.evaluate(X, y)
print("Accuracy : %.3f" % (acc*100))

predictions = (model.predict(X) > 0.5).astype(int)
rounded = [round(x[0]) for x in predictions]
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]) )
import cleverhans.tf2.attacks
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.basic_iterative_method import basic_iterative_method
fgsm_Attacks = cleverhans.tf2.attacks.fast_gradient_method.fast_gradient_method(model_fn=model, x = X, eps=0.15, norm=1)

print("Set of FGSM attacks ", fgsm_Attacks)
cw_Attacks = cleverhans.tf2.attacks.carlini_wagner_l2.carlini_wagner_l2(model_fn=model, x = X)
print("Set of CW attacks ", cw_Attacks)
bim_Attacks = cleverhans.tf2.attacks.basic_iterative_method.basic_iterative_method(model_fn=model, x = X, eps=0.15,eps_iter=0.10, nb_iter=2, norm=2)
print("Set of BIM attacks ", bim_Attacks)