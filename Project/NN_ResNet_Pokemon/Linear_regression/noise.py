import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#模拟数据
x = np.linspace(0, 10, 50)
noise = np.random.uniform(-5,5,size=50)
print(noise)
y = 5 * x + 6 + noise
#创建模型
liner = LinearRegression()
#拟合模型
liner.fit(x.reshape(-1,1),y.reshape(-1,1))
#预测
y_pred = liner.predict(x.reshape(-1,1))
plt.figure(figsize=(5,5))
plt.scatter(x,y)
plt.plot(x,y_pred, color="r")
plt.show()
# print(liner.coef_)
# print(liner.intercept_)
