import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model


house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size1 = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]

print (house_price)
print(size1)
size2 = np.array(size1).reshape(-1, 1)
print(size2)


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)


regr = linear_model.LinearRegression()
regr.fit(size2, house_price)
print("Coefficient : ", regr.coef_)
print("Intercept : ", regr.intercept_)
size_new = 1500
price = (size_new * regr.coef_) + regr.intercept_
print(price)
print(regr.predict([[size_new]]))

graph("regr.coef_ * x + regr.intercept_", range(1000, 2700))
plt.scatter(size1, house_price, color='black')
plt.xlabel("House Price")
plt.ylabel("Size of House")
plt.show()