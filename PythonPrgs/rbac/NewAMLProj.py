import random
from warnings import simplefilter

import cleverhans.tf2.attacks.fast_gradient_method
import cleverhans.tf2.attacks.carlini_wagner_l2
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder

simplefilter(action='ignore', category=FutureWarning)

def getData(fName):
    fileName = "C://PythonPrgs/csvFiles/"+fName+".csv"
    data = pd.read_csv(fileName)
    return data

def getAttributes(dataset):
    return (dataset.columns)

def modelFitPredict(mod, X_train, Y_train, X_test):
    model = mod.fit(X_train, Y_train)
    pred  = mod.predict(X_test)
    return model, pred

def modelScores(Y_test, pred):
    print("Confusion Matrix \n", confusion_matrix(Y_test, pred))
    print("Accuracy Score", accuracy_score(Y_test,pred))
    print("F-score ", f1_score(Y_test, pred))
    print("Precision : ", precision_score(Y_test, pred))
    print("Recall: ", recall_score(Y_test, pred))

def decisionTreeAlgo():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import  tree
    dtree = DecisionTreeClassifier()
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,y)
    model,pred = modelFitPredict(dtree, X_train, y_train, X_test)
    # dtree.plot_tree(clf.fit(iris.data, iris.target))
    tree.plot_tree(clf.fit(X, y))
    # modelScores(Y_test, pred)
    print("Decision Tree : ",accuracy_score(y_test, pred))
    print(confusion_matrix(y_test, pred))
    # modelScores(Y_test, pred)
    return dtree

def naiveBayesAlgo():
    from sklearn.naive_bayes import BernoulliNB
    gnb = BernoulliNB()
    gnbModel = gnb.fit(X_train, y_train)
    gnbPred = gnb.predict(X_test)
    print("Naive Bayes : ",accuracy_score(y_test, gnbPred))
    cm = confusion_matrix(y_test, gnbPred)
    print(cm)
    return gnbModel

# def featSelectionKBest(noofFeat):
#     from sklearn.feature_selection import SelectKBest, chi2
#     bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
#     fit = bestfeatures.fit(X_train,y_train)
#     dfscores = pd.DataFrame(fit.scores_)
#     # plt.bar([i for i in range(len(fit.scores_))],fit.scores_)
#     # plt.show()
#     dfcolumns = pd.DataFrame(X.columns)
#     featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
#     featureScores.columns = ['Specs','Score']   # naming the dataframe columns
    # print(featureScores.nlargest(noofFeat,'Score'))  #print 10 best features
    return bestfeatures, dfscores, featureScores

def minmaxscaling(dataset):
    minMaxScaler = MinMaxScaler().fit(dataset)
    X_scaler = minMaxScaler.transform(dataset)
    return X_scaler


d1 = getData("KddTrain_att17")
d2 = getData("KDDTestNew11m3_s")
print(d1.shape)
print(d1.head())
print(d2.shape)
print(d2.head())
d3 = d1.copy()
combined_data = pd.concat([d1, d2])

lab = LabelEncoder()
# X_enc = pd.DataFrame(d1)
# print(X_enc.head())
catCols = ['protocol_type','flag','service','class']
for i in catCols:
    combined_data[i] = lab.fit_transform(combined_data[i])
data_x = combined_data.drop('AttCat', axis=1)
data_y = combined_data.loc[:, ['AttCat']]
X_train,X_test, y_train, y_test = train_test_split(data_x, data_y,test_size=0.20, random_state=42)
X_train = minmaxscaling(X_train)
X_test = minmaxscaling(X_test)
print(X_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
DTC = DecisionTreeClassifier()
# RFC = RandomForestClassifier() #(n_estimators=25, random_state=1)
# ETC = ExtraTreesClassifier()
# # eclf = VotingClassifier(estimators=[('lr')])
# x = X_train
# y = y_train['AttCat']
# clf = DTC.fit(x,y)
# pred = clf.score(X_test, y_test)
# print("DTC (Before) Accuracy Score : ", (pred ))
# from sklearn.feature_selection import RFE
# rfe = RFE(estimator=DTC, n_features_to_select=5)
# rfe.fit(x, y)
# for i in range(x.shape[1]):
#     print("Col: %d, Selected %s, Rank: %.3f"%(i, rfe.support_[i], rfe.ranking_[i]))
#
# # clfr = RFC.fit(x,y)
# # pred = clfr.score(X_test, y_test)
# # print("RFC Accuracy Score : ", (pred ))
# # print("---------------d1-----------\n", d1.head())
# # lab = OneHotEncoder()
# # print("___", d1.columns)
# # combined_data_enc = pd.get_dummies(combined_data, columns=['protocol_type', 'flag', 'service'])
# # # print(X_enc.shape)
# # # print(X_enc.head())
# # # print(X_enc.columns)
# # # print(X_enc['class'].head())
# y = combined_data.loc[:,'AttCat']
# X = combined_data.drop(['AttCat'], axis=1)
# # # X = X_enc
# # print(X.columns)
# X_scaled = minmaxscaling(X)
# # print("Hello")
# noofFeat = 30
# print("Hello Reduced features ")
# from sklearn.feature_selection import SelectKBest, chi2
# bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
# fit = bestfeatures.fit(X, y)
# dfscores = pd.DataFrame(fit.scores_)
# print("dfScores \n", dfscores)
# dfcolumns = pd.DataFrame(X.columns)
# featureScores = pd.concat([dfcolumns, dfscores], axis=1)  # concat two dataframes for better visualization
# featureScores.columns = ['Specs', 'Score']
# feats = featureScores.loc[:15,'Specs']
# selBestFeatures = featureScores.nlargest(noofFeat,'Score')
# print("selBestFeatures\n ", selBestFeatures)
# selBestColumns = selBestFeatures.iloc[:,0].values
# print("selBestColumns : \n", selBestColumns)
# #
# # k=0
# # selColumns = []
#
# # # myCols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','num_failed_logins','logged_in','root_shell','su_attempted','num_root','num_file_creations','num_access_files','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','same_srv_rate','diff_srv_rate','role','class']
# # for i in selBestFeatures.iloc[:, 0]:
# #     selColumns.append(i)
# # print(selColumns)
# reducedDataset = pd.DataFrame(combined_data[selBestColumns],columns=selBestColumns)
# print("reduced Dataset")
# print(reducedDataset.head())
# print(reducedDataset.columns)
#
# data_x = reducedDataset
# data_y = combined_data.loc[:, ['class']]
# X_train,X_test, y_train, y_test = train_test_split(data_x, data_y,test_size=0.20, random_state=42)
# X_train = minmaxscaling(X_train)
# X_test = minmaxscaling(X_test)
# DTC = DecisionTreeClassifier()
# RFC = RandomForestClassifier() #(n_estimators=25, random_state=1)
# ETC = ExtraTreesClassifier()
# # eclf = VotingClassifier(estimators=[('lr')])
# x = X_train
# y = y_train['class']
# clf = DTC.fit(x,y)
# pred = clf.score(X_test, y_test)
# print("DTC (After) Accuracy Score : ", (pred ))
# X_scaler = minmaxscaling(X_scaled)
# from sklearn.tree import DecisionTreeClassifier
# X = X_scaler
# X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# gnbmodel = naiveBayesAlgo()
# from sklearn.feature_selection import RFE
# rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
# rfe = rfe.fit(X_train, y_train)
# selectedFeatures = pd.DataFrame({'Feature ' : list(X_train.columns), 'Ranking' : rfe.ranking_})
# print(selectedFeatures.sort_values(by='Ranking'))
# import  tensorflow as tf
#
# loss_object = tf.keras.losses.CategoricalCrossentropy()
#
# def create_adversarial_pattern(input_image, input_label):
#   with tf.GradientTape() as tape:
#     tape.watch(input_image)
#     prediction = gnbmodel(data_x)
#     loss = loss_object(data_y, prediction)
#     gradient = tape.gradient(loss, data_x)
#     signed_grad = tf.sign(gradient)
#     return signed_grad
#
# # Get the input label of the image.
# # labrador_retriever_index = 208
# # label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
# # label = tf.reshape(label, (1, image_probs.shape[-1]))
#
# perturbations = create_adversarial_pattern(data_x, data_y)
#
# # plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0,1]
#  # Get the gradients of the loss w.r.t to the input image.
# def get_imagenet_label(probs):
#   return decode_predictions(probs, top=1)[0][0]
#   # Get the sign of the gradients to create the perturbation
# # epsilons = [0, 0.01, 0.1, 0.15]
# epsilons = 0.15
# descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]
# def display_result(data_x, description):
#   _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
#   plt.figure()
#   plt.imshow(image[0]*0.5+0.5)
#   plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
#                                                    label, confidence*100))
#   plt.show()
# for i, eps in enumerate(epsilons):
#   adv_x = data_x+ eps*perturbations
#   # adv_x = tf.clip_by_value(adv_x, -1, 1)
#   display_images(adv_x, descriptions[i])