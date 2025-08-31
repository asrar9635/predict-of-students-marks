# predict-of-students-marks
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('school.csv')
a = data[['fruits']]
b = data['cars']

model = LinearRegression()
model.fit(a,b)

predict = model.predict(a)

mae = mean_squared_error(b,predict)
mse = mean_squared_error(b,predict)
sq = np.sqrt(mse)

print('mae',mae)
print('mse',mse)
print('square',sq)

plt.figure(figsize=(10,5))
plt.hist(data['fruits'],bins = 45,color = 'red',edgecolor = 'black',label = 'The Students of Marksheet')
plt.legend()
plt.title('Whole Class Students of Perfomace')
plt.xlabel('The Student Marksheet')
plt.ylabel('The Encourage of low mark')
plt.grid(True)
plt.show()
