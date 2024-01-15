'''
Description: Implementation of MIP
             Paper: https://arxiv.org/abs/2304.01717
             Github: https://github.com/amaa11/MIP
             environment: base
             replace shap with RandomForest 

Author: tanqiong
Date: 2023-06-16 22:24:38
LastEditTime: 2023-06-16 23:30:00
LastEditors: tanqiong
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# import shap
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer

def mip_score(X_train: pd.DataFrame, y_train:pd.DataFrame):
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(X_train.shape[1])])
    # if isinstance(y_train, np.ndarray):
    #     y_train = pd.DataFrame(y_train)
    
    Columns_list = list(X_train.columns.values)
    #########################################
    dicts = dict()

    # fit rf
    rf = ExtraTreesClassifier(n_estimators=100)
    rf.fit(X_train, np.squeeze(y_train))
    
    # get importance
    vals = rf.feature_importances_
    SHAP_out = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['SHAP','feature_importance_vals'])
    SHAP_out.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    SHAP_out.reset_index(inplace=True, drop=True)
    SHAP_out.index = np.arange(1, len(SHAP_out) + 1)

    while X_train.shape[1] != 1 :
        print('Iteration: ', X_train.shape[1])                          ## number of features in each iteration
        
        # fit rf
        rf = ExtraTreesClassifier(n_estimators=100)
        rf.fit(X_train,  np.squeeze(y_train))
        
        # get importance
        vals = rf.feature_importances_

        feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['Features','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
        feature_importance.reset_index(inplace=True, drop=True)
        feature_importance.index = np.arange(1, len(feature_importance) + 1)
        #print(feature_importance)
        FEATURES = feature_importance[['Features']]
        #print(FEATURES)
        print('=====================')
        for i in range(FEATURES.shape[0]):
            #print(FEATURES.iloc[i]['Features'], end = " ")
            #print(FEATURES.iloc[i].name, end = " ")
            Frac = FEATURES.iloc[i].name/X_train.shape[1]
            #print(Frac)
            #for key in dicts:
            if FEATURES.iloc[i]['Features'] in dicts:
                dicts[FEATURES.iloc[i]['Features']].append(Frac)
            else:
                dicts[FEATURES.iloc[i]['Features']] = [Frac]
            
            #dicts[FEATURES.iloc[i]['Features']] = FEATURES.iloc[i].name/X_train.shape[1]
        
        
        # identify the most informative predictor
        Top_feature = feature_importance.iloc[0]['Features']
        
        # remove the most informative predictor from train and test data
        X_train.drop([Top_feature], axis=1, inplace=True)
        X_train.reset_index(inplace=True, drop=True)

    out = {k: [sum(dicts[k])] for k in dicts.keys()}
    out = pd.DataFrame(out.items(), columns=['Modified SHAP', 'Score'])
    out['Score'] = out['Score'].str[0]
    out.sort_values(by=['Score'], inplace=True)
    out.reset_index(inplace=True, drop=True)
    out.index = np.arange(1, len(out) + 1)
    out = pd.concat([out, SHAP_out[['SHAP']]], axis=1)
    
    out.index = out['SHAP']
    
    return out['Score'].loc[Columns_list]

if __name__ == "__main__":
    # uci数据集
    # dataset = load_iris()
    # dataset = load_diabetes()
    # dataset = load_breast_cancer(as_frame=True)
    dataset = load_breast_cancer()
    features = pd.DataFrame(dataset["data"])
    # labels = pd.DataFrame(dataset["target"])
    labels = pd.DataFrame(dataset["target"])

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,random_state=666)
    out = mip_score(X_train, X_test, y_train, y_test)

    print(out)

