#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py
# author: twh
# time: 2020/10/10 10:59


# ReadMe:
# This program uses logistic regression to forecast whether there is a cat in the picture.
# Accuracy of training set : 99.5%
# Accuracy of testing set : 68.0%


import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

train_set_x_origin, train_set_y, test_set_x_origin, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_origin.reshape(train_set_y.shape[1], -1).T
test_set_x_flatten = test_set_x_origin.reshape(test_set_y.shape[1], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


class LogisticRegression(object):

    def __init__(self, train_x, test_x, train_y, test_y, iterations, learning_rate, print_flag):
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.print_flag = print_flag

    def _getall(self):
        return self.train_x, self.test_x, self.train_y, \
               self.test_y, self.iterations, self.learning_rate, self.print_flag

    @staticmethod
    def _init_w_b(dim):
        w = np.zeros((dim, 1))
        b = 0
        return w, b

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def _propagate(self,w, b, X, Y):
        m = Y.shape[1]
        A = self._sigmoid(np.dot(w.T, X) + b)
        dz = A - Y
        dw = 1 / m * np.dot(X, dz.T)
        db = 1 / m * np.sum(dz)
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        grads = {
            "dw": dw,
            "db": db
        }
        return grads, cost

    def _optimize(self, w, b, X, Y):
        train_x, train_y, test_x, test_y, iterations, learning_rate, print_flag = self._getall()
        costs = []
        for i in range(iterations):
            grads_temp, cost = self._propagate(w, b, X, Y)
            dw = grads_temp["dw"]
            db = grads_temp["db"]
            w = w - dw * learning_rate
            b = b - db * learning_rate
            if i % 100 == 0:
                costs.append(cost)
            if print_flag and i % 100 == 0:
                print("迭代次数：%i,误差值：%f" % (i, cost))

        params_opt = {
            "w": w,
            "b": b
        }

        return params_opt, costs

    def _predict(self,w, b, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        A = self._sigmoid(np.dot(w.T, X) + b)
        for i in range(m):
            if A[:, i] <= 0.5:
                Y_prediction[:, i] = 0
            elif 0.5 < A[:, i]:
                Y_prediction[:, i] = 1
        return Y_prediction

    def model(self):
        train_x, train_y, test_x, test_y, iterations, learning_rate, print_flag = self._getall()
        w, b = self._init_w_b(train_x.shape[0])
        params_opt, costs = self._optimize(w, b, train_x, train_y)
        w_opt, b_opt = params_opt["w"], params_opt["b"]
        train_y_prediction = self._predict(w_opt, b_opt, train_x)
        test_y_prediction = self._predict(w_opt, b_opt, test_x)
        print("训练集准确度为:", format(100 * (1 - np.mean(np.abs(train_y_prediction - train_y)))), "%")
        print("测试集准确度为:", format(100 * (1 - np.mean(np.abs(test_y_prediction - test_y)))), "%")
        params_final = {
            "w": w_opt,
            "b": b_opt,
            "costs": costs,
            "train_y_prediction": train_y_prediction,
            "test_y_prediction": test_y_prediction,
        }
        return params_final


if __name__ == "__main__":
    lr = LogisticRegression(train_set_x, train_set_y, test_set_x, test_set_y, 1500, 0.01, True)
    d = lr.model()
    tag = int(input("please input which picture you want to predict?"))
    y_pre = d["test_y_prediction"]
    y_pre = np.squeeze(y_pre)
    judge = "is" if y_pre[tag] == 1 else "is not"
    print("It", judge, "a cat!")
    plt.imshow(test_set_x_origin[tag])
    plt.show()
    if plt.waitforbuttonpress:
        plt.close('all')

# import time
# import numpy as np
# import matplotlib.pyplot as plt
# from lr_utils import load_dataset
# 
# train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# train_set_x_flatten = train_set_x_orig.reshape(train_set_y.shape[1], -1).T
# test_set_x_flatten = test_set_x_orig.reshape(test_set_y.shape[1], -1).T
# 
# train_set_x = train_set_x_flatten / 255
# test_set_x = test_set_x_flatten / 255
# print(train_set_x.shape)
# 
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))
# 
# 
# def initialize(dim):
#     w = np.zeros((dim, 1))
#     b = 0
#     return w, b
# 
# 
# def propagate(w, b, X, Y):
#     m = X.shape[1]
#     A = sigmoid(np.dot(w.T, X) + b)
#     dz = A - Y
#     dw = (1 / m) * np.dot(X, dz.T)
#     db = (1 / m) * np.sum(dz)
#     cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # np.log为自然对数

#     grads = {
#         "dw": dw,
#         "db": db
#     }
#     return grads, cost
# 
# 
# def optimize(w, b, X, Y, num_iterations, learning_rate, print_flag):
#     costs = []
# 
#     # 多次迭代反复更新w和b的值
#     for i in range(num_iterations):
#         grads, cost = propagate(w, b, X, Y)
#         dw = grads["dw"]
#         db = grads["db"]
#         w = w - learning_rate * dw
#         b = b - learning_rate * db
#         if i % 100 == 0:
#             costs.append(cost)
#             if print_flag:
#                 print("迭代次数为%d，误差为%f" % (i, cost))
#     params = {
#         "w": w,
#         "b": b
#     }
#     grads = {
#         "dw": dw,
#         "db": db
#     }
#     return params, grads, costs
# 
# 
# def predict(w, b, X):
#     m = X.shape[1]
#     Y_Prediction = np.zeros((1, m), dtype=np.int32)
#     A = sigmoid(np.dot(w.T, X) + b)
#     for i in range(A.shape[1]):
#         if A[:, i] > 0.5:
#             Y_Prediction[:, i] = 1
#         else:
#             Y_Prediction[:, i] = 0
# 
#     return Y_Prediction
# 
# 
# def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_flag):
#     w, b = initialize(X_train.shape[0])
#     params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_flag)
# 
#     w, b = params["w"], params["b"]
#     dw, db = grads["dw"], grads["db"]
#     Y_prediction_test = predict(w, b, X_test)
#     Y_prediction_train = predict(w, b, X_train)
# 
#     print("训练集准确度=", 100 * (1 - np.mean(np.abs(Y_train - Y_prediction_train))), "%")
#     print("测试集准确度=", 100 * (1 - np.mean(np.abs(Y_test - Y_prediction_test))), "%")
# 
#     d1 = {
#         "w": w,
#         "b": b,
#         "dw": dw,
#         "db": db,
#         "costs": costs,
#         "Y_prediction_test": Y_prediction_test,
#         "Y_prediction_train": Y_prediction_train,
#         "num_iterations": num_iterations,
#         "learning_rate": learning_rate,
#     }
#     return d1
# 
# 
# if __name__ == "__main__":
#     tic = time.time()
#     d_1 = model(train_set_x, train_set_y, test_set_x, test_set_y, 1000, 0.01, True)
#     toc = time.time()
#     # print('训练得到的w为：', '\n', d_1['w'])
#     # print('训练得到的b为：', '\n', d_1['b'])
#     # print('经过训练得到的代价函数为', '\n', d_1['costs'])
#     # print('训练集得到的预测猫标签为', '\n', d_1['Y_prediction_train'])
#     # print('测试集得到的预测猫标签为', '\n', d_1['Y_prediction_test'])
#     print('Time passes', '{:.2f}'.format(toc - tic), 's')
#     Y_prediction = d_1['Y_prediction_test']
#     Y_prediction = (Y_prediction.flatten())
#     Y_actual = np.squeeze(test_set_y)
#     print('使用逻辑回归预测的标签值为:', '\n', Y_prediction)
#     print('实际的标签值为:', '\n', Y_actual)
#     tag = int(input("please input which picture you want to forecast?"))
#     plt.imshow(test_set_x_orig[tag])
#     if Y_prediction[tag]:
#         print("it's a cat!^-^")
#     else:
#         print("it's not a cat QAQ")
#     plt.show()
