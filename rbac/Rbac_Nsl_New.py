import numpy as np
import pandas as pd
from  kmodes.kmodes import KModes
from setuptools import find_packages


from setuptools import  setup
setup(name="cleverhans",
    version="4.0.0",
    url="https://github.com/cleverhans-lab/cleverhans",
    license="MIT",
    install_requires=[
        "nose",
        "pycodestyle",
        "scipy",
        "matplotlib",
        "mnist",
        "numpy",
        "tensorflow-probability",
        "joblib",
        "easydict",
        "absl-py",
        "six",
    ],
    extras_require={
        "jax": ["jax>=0.2.9", "jaxlib"],
        "tf": ["tensorflow>=2.4.0", "tensorflow-probability", "tensorflow-datasets"],
        "pytorch": ["torch>=1.7.0", "torchvision>=0.8.0"],
    },
    packages=find_packages(),)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score, classification_report, matthews_corrcoef, confusion_matrix
from sklearn.decomposition import PCA
import  seaborn as sns
import random
from mlxtend.plotting import plot_pca_correlation_graph
import sklearn.metrics
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

# ------------------------ Writing to a file --------------------------------------
# def writeToFile(ipFile, OutFile):
#     line1 = ipFile.readline()
#     print(line1)
#     #
#     line1 = ipFile.readline()
#     print(line1)
#     outFile.write(line1)
#     id1 = ""
#     for i in dataset.values:
#         outFile.write(id1)
#         outFile.write("")
#         i.tofile(outFile, sep=",", format="%s")
#         outFile.write("\n")
        # id1 += 1
#     return
#
# ------------------------ Updating the column values --------------------------------------
#def updateCol(colValues, useCol):
# roles = {1:'admin', 2:'technician', 3:'superuser'}
# # # print(dataset.dtypes)
# for i in range(0, len(dataset)):
#     if(dataset.loc[i, 'class']=='normal'):
#         no = random.randint(1, 3)
#         dataset.loc[i, 'role'] = roles[no]
#     else:
#         dataset.loc[i, 'role'] = 'Invalid'

# ------------------------ Retrieving the data --------------------------------------
def getData(fName):
    fileName = "C://PythonPrgs/csvFiles/"+fName+".csv"
    data = pd.read_csv(fileName)
    return data

# ------------------------ Retrieving attributes of the data --------------------------------------
def getAttributes(dataset):
    return (dataset.columns)

# ------------------------ Fitting and Making predictions with the model --------------------------------------
def modelFitPredict(mod, X_train, Y_train, X_test):
    model = mod.fit(X_train, Y_train)
    pred  = mod.predict(X_test)
    return model, pred

# ------------------------ Displaying Model Scores --------------------------------------
def modelScores(Y_test, pred):
    print("Accuracy Score", accuracy_score(Y_test,pred))
    from sklearn.metrics import f1_score, confusion_matrix
    print("F-score ", f1_score(Y_test, pred))
    print("Confusion Matrix \n", confusion_matrix(Y_test, pred))

# ------------------------ Changing Roles in a column --------------------------------------
# def changeRoles(dataset, roles, col):
#     for i in dataset[col].values:
#         row = random.randint(0, 22544)
#         if (dataset.loc[row, col] == 'Invalid'):
#             no = random.randint(1,4)
#             dataset.loc[row, col] = roles[no]
#     return dataset

# ------------------------ Writing to a CSV file --------------------------------------
def writeToCsv(dataset, file):
    dataset.to_csv("C://PythonPrgs/csvFiles/" + file + ".csv")
    print("Done")

# ------------------------ Printing dataset details --------------------------------------
def prnDsDetails(dataset):
    print(dataset.shape)
    print(dataset.head())

# ------------------------ Changing Role --------------------------------------
def changeRoles(dataset):
    noofRecs = len(dataset['duration'])-1
    print(random.randint(0, noofRecs))
    for i in range(0, noofRecs):
        row = random.randint(0, noofRecs)
        if(dataset.loc[row, 'role'] == 'Invalid'):
            no = random.randint(0,3)
            dataset.loc[row, 'role'] = roles[no]
    return (dataset)

# ------------------------ Creating Class Column --------------------------------------
def createClassCol(dataset):
    dataset['class'] = ''
    for i in range(0, len(dataset['role'])):
        if dataset.loc[i,'attType'] in attackTypes:
            dataset.loc[i,'class'] = "Anomaly"
        else:
            dataset.loc[i, 'class'] = "Normal"
    return (dataset)

# ----------------------------- Logistic Regression Model ---------------------------------
def logRegAlgo():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
    logReg = LogisticRegression(solver='newton-cg', max_iter=100)
    logModel = logReg.fit(X_train, y_train)
    logPred = logReg.predict(X_test)
    # print(accuracy_score(y_test, logPred))
    print("----------------- Logistic Regression -----------------")
    acc = accuracy_score(y_test, logPred)
    misclass = 1 - acc
    print("Accuracy Score : ", acc)
    print("F-score : ", f1_score(y_test, logPred, average='macro'))
    print("Precision Score : ", precision_score(y_test, logPred, average='macro'))
    print("Recall Score : ", recall_score(y_test, logPred, average='macro'))
    cml = confusion_matrix(y_test, logPred)
    print("Confusion Matrix : \n", cml)

    sns.heatmap(cml, annot=True, fmt='g')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    plt.show()
    print("Matthews Corrcoeff : ", matthews_corrcoef(y_test, logPred))
    # print("Classification Report : ", classification_report(y_test, logPred))

# ----------------------------- Decision Tree Model ---------------------------------
def decisionTreeAlgo():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import  tree
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
    dtree = DecisionTreeClassifier(splitter='random')
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,y)
    model,pred = modelFitPredict(dtree, X_train, y_train, X_test)
    # dtree.plot_tree(clf.fit(iris.data, iris.target))
    # tree.plot_tree(clf.fit(X, y))
    # modelScores(Y_test, pred)
    # print("Decision Tree : ",accuracy_score(y_test, pred))
    # print(confusion_matrix(y_test, pred))
    # modelScores(Y_test, pred)
    print("----------------- Decision Tree -----------------")
    acc = accuracy_score(y_test, pred)
    misclass = 1 - acc
    print("Accuracy Score : ", acc)
    print("Precision Score : ", precision_score(y_test, pred,average='macro'))
    print("Recall Score : ", recall_score(y_test, pred, average='macro'))
    print("F-score : ", f1_score(y_test, pred, average='macro'))

    cmd = confusion_matrix(y_test, pred)
    print("Confusion Matrix : \n", cmd)
    sns.heatmap(cmd, annot=True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    plt.show()
    print("Matthews Corrcoeff : ", matthews_corrcoef(y_test, pred))
    # print("Classification Report : ", classification_report(y_test, pred))

    return dtree
# ----------------------------- Random Forest Model ---------------------------------
def randomForestAlgo():
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=41)
    rfModel = rf.fit(X_train, y_train)
    rfPred = rf.predict(X_test)
    model,pred = modelFitPredict(rf, X_train, y_train, X_test)
    acc = accuracy_score(y_test, pred)
    misclass = 1 - acc
    print("Random Forest : ",acc)
    cmr = confusion_matrix(y_test, pred)
    sns.heatmap(cmr, annot=True)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    plt.show()
    print(cmr)
    # modelScores(Y_test, rfPred)

# ----------------------------- Naive Bayes Model ---------------------------------
def naiveBayesAlgo():
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report
    gnb = GaussianNB()
    gnbModel = gnb.fit(X_train, y_train)
    gnbPred = gnb.predict(X_test)
    print("----------------- Naive Bayes -----------------")
    acc = accuracy_score(y_test, gnbPred)
    misclass = 1 - acc
    print("Accuracy Score : ", acc)
    print("Precision Score : ", precision_score(y_test, gnbPred, average='macro'))
    print("Recall Score : ", recall_score(y_test, gnbPred, average='macro'))
    print("F-score : ", f1_score(y_test, gnbPred, average='macro'))
    cmg = confusion_matrix(y_test, gnbPred)
    print("Confusion Matrix : \n", cmg)
    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    sns.heatmap(cmg, annot=True)

    plt.show()
    print("Matthews Corrcoeff : ", matthews_corrcoef(y_test, gnbPred))
    # print("Classification Report : \n", classification_report(y_test, gnbPred))

# ----------------------------- SVM Model ---------------------------------
def svmAlgo(kern):
    from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, precision_score, recall_score
    print("with {}".format(kern))
    svmClf = svm.SVC(kernel=kern)
    svmFit =svmClf.fit(X_train,y_train)
    svmPred = svmClf.predict(X_test)
    print("----------------- SVM -----------------")
    acc =  accuracy_score(y_test, svmPred)
    misclass = 1 - acc
    print("Accuracy Score : ",acc)
    print("Precision Score : ", precision_score(y_test,svmPred, average='macro'))
    print("Recall Score : ", recall_score(y_test, svmPred, average='macro'))
    print("F-score : ", f1_score(y_test, svmPred, average='macro'))
    cm1=confusion_matrix(y_test, svmPred)
    print("Confusion Matrix : \n", cm1)

    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(acc, misclass))
    # ax = plt.subplot()
    sns.heatmap(cm1, annot=True, fmt='g')
    plt.show()
    print("Matthews Corrcoeff : ", matthews_corrcoef(y_test,svmPred))

    # print("Classification Report : \n", classification_report(y_test, svmPred))

# ----------------------------- kNN Model ---------------------------------
def kNN():
    from sklearn.neighbors import (NeighborhoodComponentsAnalysis, KNeighborsClassifier)
    from sklearn.pipeline import Pipeline
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
    nca_pipe.fit(X_train, y_train)
    print("knn ------------ >")
    print(nca_pipe.score(X_test, y_test))





# -----------------------------K-means clustering on training data ---------------------------------
# def nearestNeigh(X_enc):
#     from sklearn.neighbors import NearestNeighbors
#     neigh = NearestNeighbors(n_neighbors=1)
#     nn = neigh.fit(X_enc)
#     print(nn)
#     a = neigh.radius_neighbors_graph(X_enc)
#     print(a)

# -----------------------------K-means clustering on training data ---------------------------------
# def kmeansAlgo():
#     from sklearn.cluster import KMeans
#     km = KMeans(n_clusters=4)
#     km.fit(X_train)
#     pred = km.predict(X_test)
#     print(accuracy_score(Y_test, pred))

# ----------------------------- Feature Selection using SelectFromModel ---------------------------------
# def featFromModel(model):
#     from sklearn.feature_selection import SelectFromModel
#     sfm = SelectFromModel(model, threshold=0.1, prefit=True)
#     X_sel = sfm.transform(X_enc)
#     impo = model.feature_importances_
#     indices = np.argsort(impo)[::]
#     # print(X_sel.shape)
#     # return X_sel

# ------------------------ To SelectKBest features from the dataset --------------------------------------
def featSelectionKBest(noofFeat):
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    bestfeatures = SelectKBest(score_func=chi2, k=noofFeat)
    fit = bestfeatures.fit(X,y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)  # concat two dataframes for better visualization
    featureScores.columns = ['Specs','Score']   # naming the dataframe columns
    return bestfeatures, dfscores, featureScores


# -------------------------- Feature Selection using RFE ----------------------------------
def featSelectionRFE():
    from sklearn.feature_selection import RFE
    rfe = RFE(estimator=logReg, step=1)
    rfe = rfe.fit(X_train, y_train)
    # # print(dataset.columns)
    selectedFeatures = pd.DataFrame({'Feature ' : list(X_train.columns), 'Ranking' : rfe.ranking_})
    print(selectedFeatures.sort_values(by='Ranking'))

# ----------------------------- Barplot ---------------------------------
# def showBarPlot():
    # sns.barplot(attCatCnt.index, attCatCnt.values, alpha = 0.9)
    # plt.title("Frequency")
    # plt.xlabel("Number of role")
    # plt.ylabel("role")
    # plt.show()

# ---------------------------- Predictions using single / series of inputs ----------------------------------
# def makePredictions():
    # for i in range(20, 30):
    #     X_new = [X_enc.loc[i, :]]
    #     y_new = gnb.predict(X_new)
    #     print("y_new : ",str(i) ,":",  y_new[0])

# --------------------------- PCA plot -----------------------------------
# def showPCAPlot():
    # from sklearn.decomposition import PCA
    # import matplotlib.pyplot as plt
    # from mlxtend.plotting import plot_pca_correlation_graph
    # # data = load_iris()
    # X, y = data.data, data.target
    # Y=X_enc.loc[:999, 'AttCat']
    # Y = y[:100]
    # plt.figure(figsize=(10,5))
    # X_pca = PCA().fit_transform(X_train)
    # plt.title('PCA - NSL-KDD dataset')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.scatter(X_pca[:100,0],X_pca[:100,1],c=Y)
    # _ = plot_pca_correlation_graph(X_train,X_train.columns)
    # plt.show()

# -------------------------- Correlation map using Seaborn ------------------------------------
# def corrMatrixHeatmap():
    # from sklearn.preprocessing import OrdinalEncoder
    # AttCatLabels = OrdinalEncoder(categories=['normal','Dos','Probe','R2L','U2R'])
    # labs, unique = pd.factorize(AttCatLabels, sort=True)
    # print(labs)
    # plt.figure(figsize=(12,10))
    # cor = dataset.corr()

    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()

# -------------------------- Standard SCaler ------------------------------------
def stdscaling(dataset):
    scaler = StandardScaler().fit(dataset)
    X_scaler = scaler.transform(dataset)
    return X_scaler

# -------------------------- MinMax SCaler ------------------------------------
def minmaxscaling(dataset):
    minMaxScaler = MinMaxScaler().fit(dataset)
    X_scaler = minMaxScaler.transform(dataset)
    return X_scaler


ker = 'linear'
dsNew = pd.read_csv("C://PythonPrgs/csvFiles/newFile24AttCatRoleClass.csv")
# print("DsNew : ", dsNew.shape)

lab = LabelEncoder()
X_enc = pd.DataFrame(dsNew)
catCols = ['protocol_type','flag','service','AttCat','attType','role','class']
for i in catCols:
    X_enc[i] = lab.fit_transform(X_enc[i])
# print()
y = X_enc.loc[:,'class']
X_enc = X_enc.drop(['id1','id','class', 'no','AttCat'], axis=1)
X = X_enc
print("With all features ...............")
X_scaler = minmaxscaling(X_enc)
X = X_scaler
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# print("X_train shape 1 : ", X_train.shape)

# naiveBayesAlgo()
# svmAlgo(ker)

# from sklearn.model_selection import cross_val_score
# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, X_train, y_train, cv = 5)
# print("CV Scores: ", scores)
X = X_enc
noofFeat = 24
bestfeatures, dfscores , featureScores = featSelectionKBest(noofFeat)
feats = featureScores.loc[:15,'Specs']
print(feats)
selBestFeatures = featureScores.nlargest(noofFeat,'Score')

# "{:.2f}".format(a_float)
print("selBestFeatures .... \n", selBestFeatures)
selBestColumns = selBestFeatures.iloc[:,0].values
V = 1
k=0
selColumns = []
for i in selBestFeatures.iloc[:, 0]:
    selColumns.append(i)
print(selColumns)
newDf = pd.DataFrame()
for i in getAttributes(X_enc):
    if i in selColumns:
        newDf[[i]] = X_enc[[i]]


# tmp = -90
# assert (tmp >= 0), "Colder temp"

print("With selected features : .............")
X_scaler = minmaxscaling(newDf)
X = X_scaler

X_train,X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# print("X_train shape reduced features : ", X_train.shape)
# from sklearn.crossvalidation import #KFold, cross_val_score

# k_fold = KFold(len(y), n_folds=10, shuffle=True, random_state=0)
# clf = <any classifier>
# print cross_val_score(clf, X, y, cv=k_fold, n_jobs=1)
# naiveBayesAlgo()
# # print("Calling SVM function....")
# svmAlgo(ker)
newSelCols = newDf.columns
dsNewOrig = dsNew.copy()
newSelDf = pd.DataFrame()
for i in dsNewOrig.columns:
    if i in newSelCols:
        newSelDf[[i]] = dsNewOrig[[i]]
newDf['attType'] = dsNewOrig['attType']
newSelDf['AttCat'] = ''
newSelDf['attType'] = dsNewOrig['attType']
myNewDf = pd.read_csv("C://PythonPrgs/csvFiles/finalFile.csv")
# print("nyNewDf shape ",  myNewDf.shape,"\n", myNewDf.head())
lab1 = LabelEncoder()
X_enc_new = pd.DataFrame(myNewDf)

catCols_new = ['flag','service','AttCat','attType','role','class']
for i in catCols_new:
    X_enc_new[i] = lab.fit_transform(X_enc_new[i])
y = X_enc_new.loc[:,'class']
X_enc_new = X_enc_new.drop(['id', 'class','AttCat'], axis=1) #'attType','class','role'
X_scaler = minmaxscaling(X_enc_new)
X = X_scaler
# X = X_enc_new

# print("X from finalFile modified Part 1 : ", X.shape)
print("_______________________ PART - I ___________________")
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=41)
# print("X_train shape : ", X_train.shape)
naiveBayesAlgo()
# svmAlgo(ker)
decisionTreeAlgo()
logRegAlgo()
# featSelectionRFE()
# randomForestAlgo()
work3 = pd.read_csv("C://PythonPrgs/csvFiles/admin3.csv")
# print("Work 3 : ", work3.shape, "\n", work3.head())
# print(work3['AttCat'].value_counts())
work3Cols = work3.columns
works3NumCols = work3._get_numeric_data().columns
work3CatCols = list(set(work3Cols)-set(works3NumCols))
X_enc_w3 = pd.DataFrame(data=work3, columns=work3Cols)
lab = LabelEncoder()
for i in work3CatCols:
    X_enc_w3[i] = lab.fit_transform(X_enc_w3[i])
# print("X_enc_w3 shape : \n", X_enc_w3.head())
y = X_enc_w3.loc[:,'AttCat']

X_enc_w3 = X_enc_w3.drop(['id','AttCat'], axis=1) #'attType','class','role'

X_scaler = minmaxscaling(X_enc_w3)
X = X_scaler
# X = X_enc_w3
print("_______________________ PART - II ___________________")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
naiveBayesAlgo()
svmAlgo(ker)
# # decisionTreeAlgo()
logRegAlgo()
# randomForestAlgo()
work4 = pd.read_csv("C://PythonPrgs/csvFiles/1Dos3.csv")
# print("Work 4 : \n", work4.shape, "\n", work4.head())
# print(work4['AttCat'].value_counts())
work4Cols = work4.columns
work4NumCols = work4._get_numeric_data().columns
work4CatCols = list(set(work4Cols)-set(work4NumCols))
X_enc_w4 = pd.DataFrame(data=work4, columns=work4Cols)
lab = LabelEncoder()
for i in work3CatCols:
    X_enc_w4[i] = lab.fit_transform(X_enc_w4[i])
# print("Work 4 shape : ", X_enc_w4.shape)
y = X_enc_w4.loc[:,'role']
# print(len(y))
X_enc_w4 = X_enc_w4.drop(['id','AttCat','role'], axis=1) #'attType','class','role'
X_scaler = minmaxscaling(X_enc_w4)
X = X_scaler
print("_______________________ PART - III ___________________")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=41)
# # print("Naive Bayes for only Dos attacks") # Accuracy : 0.3244
naiveBayesAlgo()
svmAlgo(ker)
# randomForestAlgo()
# decisionTreeAlgo()
#
logRegAlgo()