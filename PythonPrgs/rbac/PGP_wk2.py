import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# sns.set(color_codes=True)
# %matplotlib inline
# pd.set_option('display.max_columns', 100)
givendata = pd.read_csv("D://PythonPrgs/csvFiles/heart.csv")

print(givendata.head())
print(givendata.shape)
print(givendata.describe())
print(givendata.info())
scaler = StandardScaler()
scaledData = scaler.fit_transform(givendata[givendata.columns])
print(scaledData)
y = givendata['target']
X = givendata.drop(['target'], axis = 1)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from sklearn.
from sklearn.model_selection import train_test_split, cross_val_score
print("-------------------------------------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
knn_scores = []
for k in range(1, 21):
    rf_cls =  KNeighborsClassifier(n_neighbors=k)# SVC(kernel='rbf')# DecisionTreeClassifier() #(n_estimators=10)
    score = cross_val_score(rf_cls, X, y, cv=10)
    knn_scores.append(score.mean())
    # print("Score : ", k , " " , score.mean())
# sns.countplot(anime_data['target'])
correl = givendata .corr()
# plt.figure(figsize=(10,10))

plt.plot([k for k in range(1,21)], knn_scores, color='red')
for i in range(1, 21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,21)])
plt.xlabel('No. of neighbors (K)')
plt.ylabel('Scores')
plt.title("KNN Classifier scores for diff values of K")
plt.show()
# sns.heatmap(correl,annot=True, cmap='Greens')
#  'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r',
#  'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
#  'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn',
#  'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
#  'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
#  'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
#  'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r',
#  'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
#  'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'
# plt.show()
# print(anime_data['State'].mode().count())
# print(anime_data[anime_data.State == 'TX'].count())
# grp = anime_data.groupby('State')
# x = grp['Provider ID'].count()
# print("x: => \n",x)
# rating_data = pd.read_csv("D://PythonPrgs/csvFiles/rating.csv")
# print(rating_data .head())
# print(rating_data.shape)
# print(rating_data[rating_data['user_id'] == 5].rating.mean())
# mrpu = rating_data.groupby(['user_id']).mean().reset_index()
# mrpu['mean_rating'] = mrpu['rating']
# mrpu.drop(['anime_id', 'rating'], axis=1, inplace=True)
# print(mrpu.head())
# user = pd.merge(rating_data, mrpu, on=['user_id', 'user_id'])
# print(user.head())
# rating = rating_data.drop(rating_data[rating_data.rating < user.mean_rating].index)
# print(rating.head())
# cols = anime_data.columns
# print(cols)
# print(anime_data.info())
#
# corr1 = anime_data.corr()
# print(corr1)
# print(anime_data.isnull().sum())
# print(auto['cmdb_ci'].value_counts())
# sns.heatmap(corr1, annot=True)
# sns.boxplot(auto['incident_state'])
# auto['incident_state'] = auto['incident_state'].astype('str')
# print(set(auto['active']))
# print(auto['incident_state'].nunique())
# sns.barplot(auto['assignment_group'], auto['active'])
# plt.show()
# sns.distplot(auto['weight'], kde=False, rug=True)

# sns.jointplot(auto['weight'], auto['longitude'], kind='kde')
# sns.jointplot(auto['Bedroom'], auto['Bathroom'], kind='kde')
# sns.pairplot(auto[['dress_preference', 'ambience', 'transport']])
# sns.pairplot(auto[['Latitude', 'BuildingArea', 'Landsize']])
# sns.pairplot(auto)
# sns.boxplot(auto['latitude'], auto['longitude'])
# sns.boxplot(auto['Price'], auto['ParkingArea'])
# sns.countplot(auto['color'], hue=auto['activity'])
# sns.pointplot(auto['body_style'],auto['horsepower'], hue = auto['number_of_doors'])
# sns.catplot(x="fuel_type",y="horsepower", hue="number_of_doors", data=auto, kind="violin")
# sns.lmplot(y="horsepower",x="engine_size", data=auto)
# plt.show()
# Index(['userID', 'latitude', 'longitude', 'smoker', 'drink_level',
#        'dress_preference', 'ambience', 'transport', 'marital_status', 'hijos',
#        'birth_year', 'interest', 'personality', 'religion', 'activity',
#        'color', 'weight', 'budget', 'height'],
#       dtype='object')
# print(auto['JobRole'].value_counts())
# print(auto.shape)