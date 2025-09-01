import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
import math
from sklearn.datasets import load_iris
from latex import build_pdf

from IPython.core.display import Image
from IPython.display import Math, Latex
from scipy.stats import uniform

uniform.rvs(size = 10000, loc = 10, scale = 20)
fileName = load_iris()

myData_x = pd.DataFrame(fileName.data, columns=fileName.feature_names)
myData_y = pd.DataFrame(fileName.target)
noOfAttr = len(myData_x.keys())
print("No. of Attributes : ", noOfAttr)
attr = myData_x.keys()
print("Attributes : ",attr)
print("Data in the datatable : \n",myData_x.head(5))
# print("Data in the datatable (target) : \n",myData_y.head(5))
# plt.xlabel('sepal length (cm)')   # In the Scatter plot, the x-axis is labeled
# plt.ylabel('sepal width (cm)')   # In the Scatter plot, the y-axis is labeled
# plt.scatter(myData_x['sepal length (cm)'], myData_x['sepal width (cm)'], color = 'green', marker = '+')
# plt.show()
from scipy.stats import uniform
import seaborn as sns


data_uniform = uniform.rvs(size=10000, loc=10, scale=20)
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})
ax = sns.distplot(data_uniform, bins=100,kde=True,color='skyblue',hist_kws={"linewidth":15, 'alpha':1})
ax.set(xlabel='Uniform Distribution', ylabel='Frequency')
