# import logging
# import random
# from warnings import simplefilter
# import pandas as pd
# import numpy as np
#
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
# from tensorboard_plugin_wit._utils.inference_utils import get_categorical_features_to_sampling
# simplefilter(action='ignore', category=FutureWarning)
#
# fileName = "C://PythonPrgs/csvFiles/KDDTrain.csv"
# catCols = ['protocol_type','flag','service', 'class']
# df = pd.read_csv(fileName)
# # print(data.shape) ; print(data.head()) ; print(data.columns)
# # data1 = data.copy()
# # data1 = data1.drop(['class'], axis=1)
#
# ohe = OneHotEncoder() # ohe = OneHotEncoder() #
# feature_array = ohe.fit_transform(df[['protocol_type','flag','service']]).toarray()
# print(feature_array)
# print(ohe.categories_)
# feature_labels = ohe.categories_
# print(feature_labels)
# feature_labels = np.array(ohe.categories_)
# print("Feature Labels :",feature_labels)
# features = pd.DataFrame(feature_array, columns=feature_labels)
# print(features)
# # print(features.head())
# df_new = pd.concat([df, features], axis=1)
# # print(df_new.head())
# # X_enc = pd.DataFrame(data)
#
# # encoded_features = pd.get_dummies(data[catCols])
# # proValues = data['protocol_type'].value_counts()
# # flagValues = data['flag'].value_counts()
# # serviceValues = data['service'].value_counts()
# # print(proValues, flagValues, serviceValues)
# #
# # minmaxscaler = MinMaxScaler().fit(X)
# # X_scaler = minmaxscaler.transform(X)
# # X = X_scaler
# # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
# #
# # gnb = BernoulliNB()
# # modelGnb = gnb.fit(X_train, y_train)
# # gnbPred = gnb.predict(X_test)
# # print("Accuracy of Naive Bayes : ",accuracy_score(y_test, gnbPred))
# # print("Precision Score of Naive Bayes : ",precision_score(y_test, gnbPred))
# # print("Recall Score of Naive Bayes : ",recall_score(y_test, gnbPred))
# # print("F1-Score of Naive Bayes : ",f1_score(y_test, gnbPred))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import torch
# from torch import nn
# import math
# import matplotlib.pyplot as plt
# torch.manual_seed(111)
# train_data_len = 1024
# train_data = torch.zeros((train_data_len,2))
# train_data[:, 0] = 2 * math.pi * torch.rand(train_data_len)
# train_data[:, 1] =torch.sin(train_data[:, 0] )
# train_labels = torch.zeros(train_data_len)
# train_set = [(train_data[i], train_labels[i]) for i in range (train_data_len)]
# plt.plot(train_data[:, 0], train_data[:, 1], ".")
# plt.show()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from cleverhans.model import Model, CallableModelWrapper
from keras.layers import Dense, Dropout
# import cleverhans.model.
from keras.optimizers import adam_v2, rmsprop_v2
from keras.models import Sequential
# from  theano import gradient , tensor as T

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
# from cleverhans.tf2.utils import model_train, model_eval, batch_eval

# from cleverhans.attacks import fgsm, jsma
import sys
sys.path.append('C://PythonPrgs/rbacProj/cleverhans/attacks')
# import attacks
import model

from attacks import fgsm, jsma

from utils_tf import model_train, model_eval, batch_eval
from attacks_tf import jacobian_graph

sys.path.append('/PythonPrgs/rbacProj/cleverhans/attacks/')
from utils import other_classes
# from attacks import fgsm,jsma
# from attacks import fgsm, jsma
# from utils_tf import model_train, model_eval, batch_eval
# from attacks_tf import jacobian_graph


import tensorflow as tf
from tensorflow.python.platform import flags

from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,  VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC, LinearSVC

import matplotlib.pyplot as plt
plt.style.use('bmh')

FLAGS = tf.compat.v1.flags.FLAGS
flags.DEFINE_integer('nb_epochs', 20, "No. of epochs in the training model")
flags.DEFINE_integer('batch_size', 128, "Size of Training batches")
flags.DEFINE_float('learning_rate', 0.1, "Learning rate for training")
flags.DEFINE_integer('nb_classes', 5, "No. of Classification classes")
flags.DEFINE_integer('source_samples', 10, "No. of test set samples to attack ")

print()
print("Preprocessing Stage ")
names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class']
df = pd.read_csv("C://PythonPrgs/csvFiles/KDDTrain1.csv", names=names, header=None)
dft = pd.read_csv("C://PythonPrgs/csvFiles/KDDTestNew11m3.csv", names=names, header=None)
print("Initial test and training data shapes", df.shape, dft.shape)
full  = pd.concat([df, dft])
print(full.columns)
assert full.shape[0] == df.shape[0] + dft.shape[0]

full['label'] = full['class']
#
# # DoS Attacks
full.loc[full.label == 'neptune', 'label'] = 'Dos'
full.loc[full.label == 'back', 'label'] = 'Dos'
full.loc[full.label == 'land', 'label'] = 'Dos'
full.loc[full.label == 'pod', 'label'] = 'Dos'
full.loc[full.label == 'smurf', 'label'] = 'Dos'
full.loc[full.label == 'teardrop', 'label'] = 'Dos'
full.loc[full.label == 'mailbomb', 'label'] = 'Dos'
full.loc[full.label == 'processtable', 'label'] = 'Dos'
full.loc[full.label == 'udpstorm', 'label'] = 'Dos'
full.loc[full.label == 'apache2', 'label'] = 'Dos'
full.loc[full.label == 'worm', 'label'] = 'Dos'
full.loc[full.label == 'buffer_overflow', 'label'] = 'U2R'
full.loc[full.label == 'loadmodule', 'label'] = 'U2R'
full.loc[full.label == 'perl', 'label'] = 'U2R'
full.loc[full.label == 'rootkit', 'label'] = 'U2R'
full.loc[full.label == 'sqlattack', 'label'] = 'U2R'
full.loc[full.label == 'xterm', 'label'] = 'U2R'
full.loc[full.label == 'ps', 'label'] = 'U2R'
full.loc[full.label == 'ftp_write', 'label'] = 'U2R'
full.loc[full.label == 'guess_passwd', 'label'] = 'R2L'
full.loc[full.label == 'imap', 'label'] = 'R2L'
full.loc[full.label == 'phf', 'label'] = 'R2L'
full.loc[full.label == 'spy', 'label'] = 'R2L'
full.loc[full.label == 'warezmaster', 'label'] = 'R2L'
full.loc[full.label == 'warezclient', 'label'] = 'R2L'
full.loc[full.label == 'xlock', 'label'] = 'R2L'
full.loc[full.label == 'xsnoop', 'label'] = 'R2L'
full.loc[full.label == 'snmpgetattack', 'label'] = 'R2L'
full.loc[full.label == 'httptunnel', 'label'] = 'R2L'
full.loc[full.label == 'snmpguess', 'label'] = 'R2L'
full.loc[full.label == 'sendmail', 'label'] = 'R2L'
full.loc[full.label == 'named', 'label'] = 'R2L'
full.loc[full.label == 'satan', 'label'] = 'Probe'
full.loc[full.label == 'ipsweep', 'label'] = 'Probe'
full.loc[full.label == 'nmap', 'label'] = 'Probe'
full.loc[full.label == 'portsweep', 'label'] = 'Probe'
full.loc[full.label == 'saint', 'label'] = 'Probe'
full.loc[full.label == 'mscan', 'label'] = 'Probe'

# full = full.drop(['no'], axis=1)
print("Unique Labels ", full['label'].unique())
full2 = pd.get_dummies(full, drop_first=False)
features = list(full2.columns[: -5])
y_train = np.array(full2[0:df.shape[0]] [['label_normal', 'label_Dos', 'label_Probe', 'label_U2R', 'label_R2L']])
X_train = full2[0:df.shape[0]][features]
y_test = np.array(full2[df.shape[0] : ][['label_normal', 'label_Dos', 'label_Probe', 'label_U2R', 'label_R2L']]) #'label_normal', 'label_dos', 'label_probe', 'label_u2r', 'label_r2l'
X_test = full2[df.shape[0]: ][features]

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train))
X_test_scaled = np.array(scaler.transform(X_test))

# labels = full['label']
# le = LabelEncoder()
# y_full1 = le.fit(full['label'])
# y_full = le.transform(full['label'])
# y_train_1 = y_full[0:df.shape[0]]
# y_test_1 = y_full[df.shape[0]:]
# print("Training dataset shape ", X_train_scaled.shape, y_train.shape)
# print("Testing dataset shape ", X_test_scaled.shape, y_test.shape)
# print("Label encoder y shape ", y_train_1.shape, y_test_1.shape)

print("-------------------------------Done-------------------------------")
#
def mlp_model():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(FLAGS.nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def evaluate():
    eval_params = {'batch_size':FLAGS.batch_size}
    accuracy = model_eval(sess, x, y, predictions, X_test_scaled, y_test, args=eval_params)
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

# x = tf.placeholder(tf.float32, shape = (None, X_train_scaled.shape[0]))
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2)) # (shape=[None, 2], dtype=tf.float32)
y = tf.compat.v1.placeholder(dtype=tf.float32, shape = (None, FLAGS.nb_classes))
# y = tf.compat.v1.placeholder(dtype=float32, shape=(None, FLAGS.nb_classes) )

tf.set_random_seed(42)
model = mlp_model()
sess = tf.Session()
predictions = model(x)
init = tf.global_variables_initializer()
sess.run(init)
FLAGS = tf.compat.v1.flags.FLAGS
train_params = {'nb_epochs': FLAGS.nb_epochs,
                        'batch_size':FLAGS.batch_size,
                        'learning_rate':FLAGS.learning_rate,
                        'verbose':0}
model_train(sess, x, y, predictions, X_train_scaled, y_train, evaluate=evaluate, args=train_params)
source_samples = X_test_scaled.shape[0]
results = np.zeros((FLAGS.nb_classes, source_samples), dtype='i')
perturbations = np.zeros((FLAGS.nb_classes, source_samples), dtype='f')
grads = jacobian_graph(predictions, x, FLAGS.nb_classes)

X_adv = np.zeros((source_samples, X_test_scaled.shape[1]))
for sample_ind in range (0, source_samples):
    current_class = int(np.argmax(y_test[sample_ind]))
    for target in [0]:
        if current_class == 0:
            break
        adv_x, res, percent_perturb = jsma(sess, x, predictions, grads,X_test_scaled [ sample_ind : ( sample_ind+1) ],target , theta =1 , gamma =0.1 ,increase = True , back ='tf', clip_min = 0, clip_max = 1)

        X_adv[sample_ind] = adv_x
        results[target, sample_ind] = res
        perturbations[target, sample_ind] = percent_perturb
print(X_adv.shape)
print (" -------------- Evaluation of MLP performance -- ---- ----- ---")
eval_params = {'batch_size ': FLAGS.batch_size}
accuracy = model_eval(sess, x,y, predictions, X_test_scaled, y_test, args=eval_params)
print("Test accuracy on normal examples: ", str(accuracy))

print()
print (" -------------- Decision Tree Classifier -- ---- ----- ---")
dt = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)
fpr_dt, tpr_dt , _  = roc_curve(y_test[:, 0], y_pred[:, 0])
roc_auc_dt = auc(fpr_dt, tpr_dt)
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred, average='micro'))
print("AUC Score : ", roc_auc_dt)

y_pred_adv = dt.predict(X_adv)
fpr_dt_adv, tpr_dt_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_dt_adv = auc(fpr_dt_adv, tpr_dt_adv)
print("Accuracy Score Adversarial : ", accuracy_score(y_test, y_pred_adv))
print("F1 Score Adversarial : ", f1_score(y_test, y_pred_adv, average='micro'))
print("AUC Score Adversarial: ", roc_auc_dt_adv)
plt.figure()
lw = 2
plt.plot(fpr_dt, tpr_dt, color= 'darkorange', lw = lw, label='ROC Curve(area = %0.2f)' %roc_auc_dt)
plt.plot(fpr_dt_adv, tpr_dt_adv, color= 'green', lw = lw, label='ROC Curve adv (area = %0.2f)' %roc_auc_dt_adv)
plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate ")
plt.ylabel("True Positive Rate")
plt.title("ROC Decision Tree (class= Normal)")
plt.legend(loc="lower right")
plt.savefig("ROC_DT.png")


feats = dict()
total = 0
orig_attack = X_test_scaled - X_adv
for i in range (0, orig_attack.shape[0]):
    ind = np.where(orig_attack[i, :] != 0)[0]
    total += len(ind)
    for j in ind:
        if j in feats:
            feats[j] += 1
        else:
            feats[j] = 1
print("No. of unique features changed ", len(feats.keys()))
print("NO. of avg features changed per datapoint ", total/len(orig_attack))
top_10 = sorted(feats, key=feats.get, reverse=True)[:10]
top_20 = sorted(feats, key=feats.get, reverse=True)[:20]
print("Top ten features ", X_test.columns[top_10])

top_10_val = [100* feats [k] / y_test.shape[0] for k in top_10]
top_20_val = [100* feats [k] / y_test.shape[0] for k in top_20]

plt.figure(figsize=(16, 12))
plt.bar(np.arange(20), top_20_val, align='center')
plt.xticks(np.arange(20), X_test.columns[top_20], rotation = 'vertical')
plt.title("Features participation in adversarial samples ")
plt.ylabel("Percentage ")
plt.xlabel("Features")
plt.savefig("Adv_features.png")
adv_x_f = fgsm(x, predictions,eps = 0.3)
X_test_adv, = batch_eval(sess, [x], [adv_x_f], [X_test_scaled])

accuracy = model_eval(sess, x, y, predictions, X_test_adv, y_test)
print("Test accuracy on adv sample" + str(accuracy))
feats = dict()
total = 0
orig_attack = X_test_scaled - X_test_adv
for i in range (0, orig_attack.shape[0]):
    ind = np.where(orig_attack[i, :] != 0)[0]
    total += len(ind)
    for j in ind:
        if j in feats:
            feats[j] += 1
        else:
            feats[j] = 1
print("No. of unique features changed with FGSM ", len(feats.keys()))
print("No. of avg features changed per datapoint with FGSM", total/len(orig_attack))