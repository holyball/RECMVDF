'''
Description: Implementation of NMR
             Paper: https://ieeexplore.ieee.org/document/9897253
             Github: https://github.com/amaa11/NMR
             environment: base
Author: tanqiong
Date: 2023-06-16 22:26:29
LastEditTime: 2023-06-16 22:56:31
LastEditors: tanqiong
'''
import pandas as pd
import shap
from statistics import mean
from sklearn.linear_model import LogisticRegression
from numpy import mean
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer


# uci数据集
dataset = load_breast_cancer(as_frame=True)
features = pd.DataFrame(dataset["data"])
labels = pd.DataFrame(dataset["target"])

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=666
)

Columns_list = list(X_train.columns.values)
dicts = dict()
m = 0
MPM_list = []       # 可能的排名变化最大值
while X_train.shape[1] != 1:
    MPM = 0
    NF = X_train.shape[1]
    for j in range(1, NF, 2):
        summ = NF-j
        MPM = MPM+summ
    MPM = MPM*2
    MPM_list.append(MPM)
    #print('MPM :', MPM)

    #print('Iteration: ', X_train.shape[1])                          ## number of features in each iteration
    model = LogisticRegression(C=0.1)  # defin a model
    model.fit(X_train, y_train.values.ravel())  # fit the data to the model
    ypred_test = model.predict(X_test)  # predict test data

    #prepare the data for shap
    masker = shap.maskers.Independent(data=X_train)

    # fit shap
    explainer = shap.LinearExplainer(model, masker=masker)

    # apply shap to test data
    shap_values = explainer.shap_values(X_test)

    # the follwoing steps to represents the SHAP outcome (informative predictors) as datafram
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)), columns=[
                                      'Features', 'feature_importance_vals'])
    feature_importance.sort_values(
        by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.reset_index(inplace=True, drop=True)
    feature_importance.index = np.arange(1, len(feature_importance) + 1)
    #print(feature_importance)
    FEATURES = feature_importance[['Features']]
    #print(FEATURES)
    for i in range(FEATURES.shape[0]):
        #print(FEATURES.iloc[i]['Features'], end = " ")
        #print(FEATURES.iloc[i].name, end = " ")
        #for key in dicts:
        if FEATURES.iloc[i]['Features'] in dicts:
            dicts[FEATURES.iloc[i]['Features']].append(
                FEATURES.iloc[i].name + m)
        else:
            dicts[FEATURES.iloc[i]['Features']] = [FEATURES.iloc[i].name + m]

    m = m+1
    #print('\t')
    # identify the most informative predictor
    Top_feature = feature_importance.iloc[0]['Features']

    # remove the most informative predictor from train and test data
    X_train.drop([Top_feature], axis=1, inplace=True)
    X_train.reset_index(inplace=True, drop=True)
    X_test.drop([Top_feature], axis=1, inplace=True)
    X_test.reset_index(inplace=True, drop=True)
    #print('===================================================')
MPM_list.pop(0)

lenOfLists=len(dicts)
#function to add item at index=len ... append item
def add_item_to_dict(my_dict, key, value):
    my_dict[len(my_dict)] = [key,value]

    
sorted_dict = {}

for index in range(1,lenOfLists):
    for key, value in dicts.items():
        if len(value)==index:
            add_item_to_dict(sorted_dict,key,value)

#calculate sum of changes
sumOfChanges={}
#first is zero
add_item_to_dict(sumOfChanges,sorted_dict[0][0],0)

for index in range(1,lenOfLists-1):  
    key=sorted_dict[index][0]
    soc=0   # 每一次迭代后，特征重要性排名的变换的总和
    for t in range(index,lenOfLists):        
        soc+=abs(sorted_dict[t][1][index]-sorted_dict[t][1][index-1])
        #print(sorted_dict[t][1][index],sorted_dict[t][1][index-1],(sorted_dict[t][1][index]-sorted_dict[t][1][index-1]))
    add_item_to_dict(sumOfChanges,key,soc)
    #print(soc,'----') 
#last is same to last -1
add_item_to_dict(sumOfChanges,sorted_dict[lenOfLists-1][0],sumOfChanges[lenOfLists-2][1])
#print(sorted_dict)
#print(sumOfChanges)
sumOfChanges.popitem()
first_key = next(iter(sumOfChanges))
del sumOfChanges[first_key]
rates = []
ite=0
for key, value in sumOfChanges.items():    
    rates.append(value[1]/MPM_list[ite])
    ite=ite+1
#print(rates)
print("NMR value : ", round(mean(rates), 3))