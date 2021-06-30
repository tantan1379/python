from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读入数据集
diabetes = datasets.load_diabetes()  # diabetes为有三个key的字典
# 拆分数据
data = diabetes['data']
target = diabetes['target']
feature_names = diabetes['feature_names']
# print(data.shape)
# print(target.shape)
# print(feature_names)
print(data)
df = pd.DataFrame(data, columns=feature_names)
# print(df.head())
# print(df.info())
train_X, test_X, train_Y, test_Y = train_test_split(
    data, target, train_size=0.8, test_size=0.2)
model = LinearRegression()
model.fit(train_X,train_Y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

# plt.figure(figsize=(12,25))
# for i,col in enumerate(df.columns):
#     train_X = df.loc[:,col].reshape(-1,1)