import pandas as pd
import tensorflow as tf
import pandas as pd
from sklearn import metrics
from scipy.stats import zscore
import os

df = pd.read_csv("C://PythonPrgs/csvFiles/KddTrain_att17.csv")
print(df.head())


def expand_categories(values):
    result = []
    s = values.value_counts()
    t = float(len(values))
    for v in s.index:
        result.append("{}:{}%".format(v, round(100 * (s[v] / t), 2)))
    return "[{}]".format(",".join(result))


def analyze(df):
    print()
    cols = df.columns.values
    total = float(len(df))

    print("{} rows".format(int(total)))
    for col in cols:
        uniques = df[col].unique()
        unique_count = len(uniques)
        if unique_count > 100:
            print("** {}:{} ({}%)".format(col, unique_count, \
                                          int(((unique_count) / total) * 100)))
        else:
            print("** {}:{}".format(col, expand_categories(df[col])))
            expand_categories(df[col])

analyze(df)


# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd


# Encode text values to dummy variables(i.e. [1,0,0],
# [0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

# Now encode the feature vector

pd.set_option('display.max_columns', 6)
pd.set_option('display.max_rows', 5)

for name in df.columns:
  if name == 'attCat':
    pass
  elif name in ['protocol_type','service','flag', 'class'] : #,'land','logged_in','is_host_login','is_guest_login']:
    encode_text_dummy(df,name)
  else:
    encode_numeric_zscore(df,name)

# display 5 rows

df.dropna(inplace=True,axis=1)
df[0:5]


# Convert to numpy - Classification
x_columns = df.columns.drop('attCat')
x = df[x_columns].values
dummies = pd.get_dummies(df['attCat']) # Classification
# from sklearn.preprocessing import LabelEncoder

# classLab = LabelEncoder()
# df['class'] = classLab.fit_transform(df['class'])
# print(df['class'].unique())
outcomes = dummies.columns
num_classes = len(outcomes)
# y = df['class'].values
y = dummies.values
print(df.columns)
print(y.shape)


print(df.groupby('attCat')['attCat'].count())
print(df['attCat'].values)


from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers

# Create a test/train split.  20% test
# Split into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# # Create neural net
model = Sequential()
model.add(layers.Dense(20))
model.add(layers.Dense(50))
model.add(layers.Dense(20))
model.add(Dense(1, kernel_initializer='normal'))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
                        patience=5, verbose=1, mode='auto',
                          restore_best_weights=True)
model.fit(x,y,validation_data=(x_test,y_test),
          callbacks=[monitor],verbose=2,epochs=10)

from sklearn import model_selection
# Measure accuracy
import numpy as np
pred = model.predict(x_test)
pred = np.argmax(pred,axis=1)
y_eval = np.argmax(y_test,axis=1)
accScore = metrics.accuracy_score(y_eval, pred)
print("* * * * * * * * Keras Model Metrics * * * * * * * * ")
print("Validation score: {}".format(accScore))
precScore = metrics.precision_score(y_eval, pred, average='macro')
print("Precision score: {}".format(precScore))
recScore = metrics.recall_score(y_eval, pred, average='macro')
print("Recall score: {}".format(recScore))
f1Score = metrics.f1_score(y_eval, pred, average='macro')
print("F1 score : {}".format(f1Score))
confMat = metrics.confusion_matrix(y_eval, pred)
print("Confusion Matrix : ")
print(confMat)
print("Classification Report")
print(metrics.classification_report(y_eval, pred))

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow as tf
from art.estimators.classification import KerasClassifier, TensorFlowClassifier
classifier = KerasClassifier(model=model, use_logits=False)
print(classifier)

classifier.fit(x_train, y_train,batch_size=10,nb_epochs=10)