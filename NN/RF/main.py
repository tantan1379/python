#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# file: main.py.py
# author: twh
# time: 2021/3/26 8:53
from sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
data=[]
traffic_feature=[]
traffic_target=[]
csv_file = csv.reader(open('packSize_all.csv'))
for content in csv_file:
    content=list(map(float,content))
    if len(content)!=0:
        data.append(content)
        traffic_feature.append(content[0:6])
        traffic_target.append(content[-1])
print('data=',data)
print('traffic_feature=',traffic_feature)
print('traffic_target=',traffic_target)
scaler = StandardScaler() # 标准化转换
scaler.fit(traffic_feature)  # 训练标准化对象
traffic_feature= scaler.transform(traffic_feature)   # 转换数据集
feature_train, feature_test, target_train, target_test = train_test_split(traffic_feature, traffic_target, test_size=0.3,random_state=0)
# clf = RandomForestClassifier(criterion='entropy')
clf = RandomForestClassifier()
clf.fit(feature_train,target_train)
predict_results=clf.predict(feature_test)
print(accuracy_score(predict_results, target_test))
conf_mat = confusion_matrix(target_test, predict_results)
print(conf_mat)
print(classification_report(target_test, predict_results))