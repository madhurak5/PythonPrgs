
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
dataframe = pandas.read_csv("c://PythonPrgs/csvFiles/Files/iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

"""
When modeling multiclass classification problems using neural networks, it is good practice to 
reshape the output attribute from a vector that contains values for each class value to be a 
matrix with a boolean for each class value and whether or not a given instance has that class 
value or not. This is called one hot encoding or creating dummy variables from a categorical variabl
"""
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
"""
Fully connected network with one hidden layer which contains 4 neurons.
The hidden layer uses a rectifier activation function which is a good practice. Because we used 
a one-hot encoding for our iris dataset, the output layer must create 3 output values, one for 
each class. The output value with the largest value will be taken as the class predicted by the model. 
"""
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(4, kernel_initializer= 'normal' , activation= 'relu' ,input_dim=4))
    model.add(Dense(3, kernel_initializer= 'normal' , activation= 'softmax' ))
    # Compile model
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=15, verbose=0)

# Define the model evaluation procedure
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))