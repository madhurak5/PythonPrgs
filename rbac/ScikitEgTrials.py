import numpy as np
from sklearn import preprocessing
X_train = [[1, -1, 2],
           [2, 0, 0],
           [0, 1, -1]
           ]
X_scaled = preprocessing.scale(X_train)
print("Scaled X_train: \n", X_scaled)
print("Mean: ", X_scaled.mean(axis=0))
print("Variance: ", X_scaled.std(axis=0))
scaler = preprocessing.StandardScaler().fit(X_train)
print("Scaler: ", scaler)
print("Scaler scale: ", scaler.scale_)
print("Transformed scaler on X_train: \n", scaler.transform(X_train))
X_test = [[-1, 1, 0]]
print("Transformed scaler on X_test: ", scaler.transform(X_test))
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
print("MinMaxScaler on X_train: \n", X_train_minmax)
X_test = np.array([[-3., -1.,  4.]])
X_test_minmax = min_max_scaler.transform(X_test)
print("minMaxScaler on X_test: ", X_test_minmax)
max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
print("MaxAbsScaler on X_train: ", X_train_maxabs)
X_test = np.array([[ -3., -1.,  4.]])
X_test_maxabs = max_abs_scaler.transform(X_test)
print("MaxAbsScaler on X_test: ", X_test_maxabs)
print("Scale: ", max_abs_scaler.scale_)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans = quantile_transformer.transform(X_test)
print("Percentile on X_train: ", np.percentile(X_train[:, 0], [0, 25, 50, 75, 100]))
print("Percentile on X_train_trans: ", np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))
print("Percentile on X_test: ",np.percentile(X_test[:, 0], [0, 25, 50, 75, 100]))
print("Percentile on X_test_trans: ", np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
print("Encoder: " ,enc.fit(X))
print("After transformation: ", enc.transform([['female', 'from US', 'uses Firefox']]))
enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
print("Encoder: ",enc.fit(X))
print("Transformed: \n", enc.transform([['female', 'from US', 'uses Safari'],['male', 'from Europe', 'uses Safari']]).toarray())
print("Categories: \n", enc.categories_)
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
print("Encoder: ", enc.fit(X))
print("Transformed: \n", enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray())

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')

plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("D://Pythonprgs/csvFiles/KddTrain_att17.csv")
print(data.head())
dataset_lab = LabelEncoder()
data = data.apply(dataset_lab.fit_transform)
X = data.drop('AttCat', axis=1)
y = data['AttCat']
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
const_filter = VarianceThreshold(threshold=0.01)
const_filter.fit(X_train)
X_train_filter = const_filter.transform(X_train)
X_test_filter = const_filter.transform(X_test)
X_train_T = X_train_filter.T
X_test_T = X_test_filter.T
X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)
print(X_train_T.duplicated().sum())

duplicated_features = X_train_T.duplicated()
print("Duplicated features : ", duplicated_features)
featuresToKeep = [not index for index in duplicated_features]
print("Features to keep : ", len(featuresToKeep))
X_train_unique = X_train_T[featuresToKeep].T
X_test_unique = X_test_T[featuresToKeep].T
X_train_unique = pd.DataFrame(X_train_unique)
X_test_unique = pd.DataFrame(X_test_unique)
print(X_train_unique.shape, X_test_unique.shape)
corrmat = X_train_unique.corr()
def get_correlation(data, threshold):
    corr_col = set()
    corrmat = data.corr()
    for i in range (len(corrmat.columns)):
        for j in range(i):
            if (abs(corrmat.iloc[i, j]) > threshold):
                colName = corrmat.columns[i]
    return corr_col

corr_features = get_correlation(X_train_unique, 0.70)

print("Correlated Features : ", len(set(corr_features)))
print(corr_features)
X_train_uncorr = X_train_unique.drop(labels=corr_features, axis=1)
X_test_uncorr = X_test_unique.drop(labels=corr_features, axis=1)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=4)
X_train_lda = lda.fit_transform(X_train_uncorr, y_train)
print("Shape of X_train lda : ", X_train_lda.shape)
X_test_lda = lda.transform(X_test_uncorr)
print("Shape of X_test lda : ", X_test_lda.shape)

def run_RF(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy Score : ",accuracy_score(y_test, y_pred))


run_RF(X_train_lda, X_test_lda, y_train, y_test)
run_RF(X_train, X_test, y_train, y_test)


from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=41)
pca.fit(X_test_uncorr)
X_train_pca = pca.transform(X_train_uncorr)
X_test_pca = pca.transform(X_test_uncorr)
print(X_train_pca.shape)
print(X_test_pca.shape)

run_RF(X_train_pca, X_test_pca, y_train, y_test)
run_RF(X_train, X_test, y_train, y_test)

for comp in range (1, 40):
    pca = PCA(n_components=comp, random_state=41)
    pca.fit(X_test_uncorr)
    X_train_pca = pca.transform(X_train_uncorr)
    X_test_pca = pca.transform(X_test_uncorr)
    print("Selected components : ", comp)
    run_RF(X_train_pca, X_test_pca, y_train, y_test)
    print()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest, RFE
dataset = pd.read_csv("D://PythonPrgs/csvFiles/KDDTrain.csv")
sns.set(color_codes=True)
sns.set(style="ticks", color_codes=True)
x = np.random.normal(size=100)
sns.catplot(x="protocol_type", y="duration", data=dataset)
sns.distplot(x)
sns.distplot(x, bins=20, kde=False, rug=True)
sns.distplot(x, hist=False, rug=True)
sns.pairplot(dataset)
plt.show()
print(dataset.head())
dataLab = LabelEncoder()
dataset = dataset.apply(dataLab.fit_transform)
# print(dataset.head())
# print(dataset['class'].value_counts())
y = dataset.iloc[:,-1]
X = dataset.iloc[:,0:-1]
X_std = StandardScaler()
X_Norm = preprocessing.Normalizer()
norm = X_Norm.fit(dataset)
scaler = preprocessing.StandardScaler().fit(dataset)
encoding = preprocessing.OneHotEncoder(sparse=False)
dataset = encoding.fit_transform(dataset)
y = dataset[:, 0:1]
X = dataset[:,1:23]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
from keras.layers import  Dense
from keras.models import Sequential
import keras.utils
model = Sequential()
model.add(Dense(1, input_dim=22, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=25)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
gnb = GaussianNB()
gnbModel = gnb.fit(X_train, y_train)
gnbPred = gnbModel.predict(X_test)
gnbScore = accuracy_score(y_test, gnbPred)
print("Accuracy Score of GaussianNB : ", gnbScore)
dtree = DecisionTreeClassifier()
dtreeModel = dtree.fit(X_train, y_train)
dtreePred = dtreeModel.predict(X_test)
dtreeScore = accuracy_score(y_test, dtreePred)
cvDtreeScore = cross_val_score(dtreeModel, X, y, cv = 4)
print("Accuracy Score of Decision Tree: ", dtreeScore )
print("CV Score of Decision Tree: ", cvDtreeScore)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, random_state=41)
rfcModel = rfc.fit(X_train, y_train)
rfcPred= rfc.predict(X_test)
rfcScore = accuracy_score(y_test, rfcPred)
print("Accuracy Score of Random Forest Classifier : ", rfcScore)
from sklearn.svm import SVC
svc =SVC(kernel='linear')
svcModel = svc.fit(X_train, y_train)
svcPred = svcModel.predict(X_test)
svcScore = accuracy_score(y_test, svcPred)
print("Accuracy Score of SVC Classifier : ", svcScore)
from sklearn.neighbors import KNeighborsClassifier
kNNclf = KNeighborsClassifier(n_neighbors=2).fit(X_train, y_train)
knnPred = kNNclf.predict(X_test)
knnScore = accuracy_score(y_test, knnPred)
print("Accuracy Score of KNN Classifier : ", knnScore)