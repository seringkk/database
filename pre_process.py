"""
*描述: 对数据进行预处理的函数
*File: pre_process.py
* ======★★★
* @作者: 星队 
* ======★★★
*创建日期: 2022/11/9 17:20 
"""
import pandas as pd
import sys
import os
import numpy as np
import pandas as pd

def processing(data):

    data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']]  # 筛选需要的列
    data['Age'] = data['Age'].fillna(data['Age'].mean())    # 平均值填充
    data['Cabin'] = pd.factorize(data.Cabin)[0]    # 数值化
    data.fillna(0, inplace=True)    # 剩余的缺失值用0填充
    data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]  # 男变为 1, 女变为 0

    def trans_p(p, num):
        data[p] = np.array(data['Pclass'] == num).astype(np.int32)
    trans_p('p1', 1)
    trans_p('p2', 2)
    trans_p('p3', 3)
    del data['Pclass']     # 删除此列

    def trans_e(e, ch):
        data[e] = np.array(data['Embarked'] == ch).astype(np.int32)
    trans_e('e1', 'S')
    trans_e('e2', 'C')
    trans_e('e3', 'Q')
    del data['Embarked']

    data_train = data[['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'p1', 'p2','p3', 'e1', 'e2', 'e3']]
    data_target = data['Survived'].values.reshape(len(data),1)
    # print(data_train)
    # print(data_target)

    # return data
    return [data_train, data_target]


# 对测试集进行处理
def pre_pro_test(data):
    data = data[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin', 'Embarked']]  # 筛选需要的列
    data['Age'] = data['Age'].fillna(data['Age'].mean())    # 平均值填充
    data['Cabin'] = pd.factorize(data.Cabin)[0]    # 数值化
    data.fillna(0, inplace=True)    # 剩余的缺失值用0填充
    data['Sex'] = [1 if x == 'male' else 0 for x in data.Sex]  # 男变为 1, 女变为 0

    def trans_p(p, num):
        data[p] = np.array(data['Pclass'] == num).astype(np.int32)
    trans_p('p1', 1)
    trans_p('p2', 2)
    trans_p('p3', 3)

    def trans_e(e, ch):
        data[e] = np.array(data['Embarked'] == ch).astype(np.int32)
    trans_e('e1', 'S')
    trans_e('e2', 'C')
    trans_e('e3', 'Q')

    del data['Pclass']
    del data['Embarked']
    return data