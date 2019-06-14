import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

salary_data = pd.read_csv("salary.csv")
print(salary_data.describe())
print(salary_data.head(3))
#draw a scatterpoints picture
plt.scatter(salary_data.AvgSalary,salary_data.AlgoSalary) 

print(salary_data['AvgSalary'].values)
x = np.reshape(salary_data['AvgSalary'].values,newshape=(len(salary_data['AvgSalary']),1))
y = np.reshape(salary_data['AlgoSalary'].values,newshape=(len(salary_data['AlgoSalary']),1))

lr = LinearRegression()
lr.fit(x,y)
# Returns the coefficient of determination R^2 of the prediction
print(lr.score(x,y))

y_predict = lr.predict(x)
plt.plot(x,y_predict)
plt.show()

