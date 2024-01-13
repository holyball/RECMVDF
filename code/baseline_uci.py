'''
Description: UCI数据集baseline, 基于sklearn v022
             CascadeForest, CatBoost需要切换到sklearn V122的环境下运行
Author: tanqiong
Date: 2023-04-04 16:12:13
LastEditTime: 2023-05-17 22:19:32
LastEditors: tanqiong
'''
# 完整数据集的baseline
import sys 
import os
import joblib
import copy
import joblib
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from deepforest import CascadeForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import joblib

import warnings
warnings.filterwarnings("ignore")

import copy
import numpy as np
import pandas as pd
from scores import scores_clf
from load_uci_data import load_my_data, my_one_hot_numpy

def get_baseline_with_cv(x, y, cv_list, result_path):
    model_list = [
        AdaBoostClassifier(n_estimators=500,),
        # ExtraTreesClassifier(n_jobs=-1, bootstrap=False, max_samples=0.7), 
        ExtraTreesClassifier(n_estimators=500, n_jobs=-1),
        RandomForestClassifier(n_estimators=500, n_jobs=-1), 
        GradientBoostingClassifier(n_estimators=500, random_state=66666), 
        # CascadeForestClassifier(n_estimators=500), 
        XGBClassifier(n_estimators=500, eval_metric=['logloss','auc','error']), 
        # CatBoostClassifier(n_estimators=500),
        LGBMClassifier(n_estimators=500),
        # MLPClassifier(hidden_layer_sizes=(62,154,274,343,258,)),
    ]

    model_name_list = [item.__class__.__name__ for item in model_list]
    metric_name_list = ["AUROC", "AUPR", 'F1', 'ACC', 'Precision', 'Recall']
    result_detail_dict = {}
    result_df = pd.DataFrame(columns=model_name_list, index=metric_name_list)
    for model in model_list:
        cv_score_list = []
        model_not_fit = copy.deepcopy(model)
        for i, (train_idx, test_idx) in enumerate(cv_list):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            if model.__class__.__name__ in ['LogisticRegression', 'RidgeClassifier', 'LinearSVC']:
                ss = StandardScaler()
                x_train = ss.fit_transform(x_train)
                x_test = ss.transform(x_test)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)
             
            predict_dir = os.path.join(result_path, model.__class__.__name__)
            if not os.path.exists(predict_dir): 
                os.mkdir(predict_dir)
            pd.DataFrame(
                np.hstack([y_test.reshape((-1, 1)), y_pred.reshape((-1, 1)), y_pred_proba]),
                columns=['y_true', 'y_pred','class_0', 'class_1'],
            ).to_csv(os.path.join(predict_dir, f"cv_{i}_{model.__class__.__name__}.csv"))

            scores = scores_clf(y_test, y_pred_proba).values
            cv_score_list.append(scores)
            del model
            model = copy.deepcopy(model_not_fit)

        result_detail_dict[model.__class__.__name__] = cv_score_list
        result_df[model.__class__.__name__] = np.mean(cv_score_list, axis=0).reshape(-1)
        result_df[model.__class__.__name__+"_std"] = np.std(cv_score_list, axis=0).reshape(-1)
    return result_detail_dict, result_df

def get_baseline_with_repeat(x_train, y_train, x_test, y_test, n_repeat, result_path):
    """在给定的训练集和测试集上重复实验"""
    n_repeat = 5
    model_list = [
        AdaBoostClassifier(n_estimators=500,),
        # ExtraTreesClassifier(n_jobs=-1, bootstrap=False, max_samples=0.7), 
        ExtraTreesClassifier(n_estimators=500, n_jobs=-1),
        RandomForestClassifier(n_estimators=500, n_jobs=-1), 
        GradientBoostingClassifier(n_estimators=500), 
        CascadeForestClassifier(n_trees=500), 
        XGBClassifier(n_estimators=500, eval_metric=['logloss','auc','error']), 
        CatBoostClassifier(n_estimators=500),
        LGBMClassifier(n_estimators=500),
        # MLPClassifier(hidden_layer_sizes=(62,154,274,343,258,)),
    ]

    model_name_list = [item.__class__.__name__ for item in model_list]
    metric_name_list = ["AUROC", "AUPR", 'F1', 'ACC', 'Precision', 'Recall']
    result_detail_dict = {}
    result_df = pd.DataFrame(columns=model_name_list, index=metric_name_list)
    random_states = np.arange(20, 20+n_repeat)
    for model in model_list:
        cv_score_list = []
        model_not_fit = copy.deepcopy(model)
        # for i, (train_idx, test_idx) in enumerate(cv_list):
        for i, random_state in enumerate(random_states):
            model.set_params(**{'random_state':random_state})
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)
             
            predict_dir = os.path.join(result_path, model.__class__.__name__)
            if not os.path.exists(predict_dir): 
                os.mkdir(predict_dir)
            pd.DataFrame(
                np.hstack([y_test.reshape((-1, 1)), y_pred.reshape((-1, 1)), y_pred_proba]),
                columns=['y_true', 'y_pred','class_0', 'class_1'],
            ).to_csv(os.path.join(predict_dir, f"cv_{i}_{model.__class__.__name__}.csv"))

            scores = scores_clf(y_test, y_pred_proba).values
            cv_score_list.append(scores)
            del model
            model = copy.deepcopy(model_not_fit)

        result_detail_dict[model.__class__.__name__] = cv_score_list
        result_df[model.__class__.__name__] = np.mean(cv_score_list, axis=0).reshape(-1)
        result_df[model.__class__.__name__+"_std"] = np.std(cv_score_list, axis=0).reshape(-1)
    return result_detail_dict, result_df
if __name__ == "__main__":
    # 加载数据

    x_train, y_train, x_test,y_test = load_my_data('adult')
    
    result_path = "../result/uci_baseline_result_sklearn_V122_n_estimator_500"
    if not os.path.exists(result_path): os.mkdir(result_path)
    report_detail_dict, result_df = get_baseline_with_repeat(x_train, y_train, x_test,y_test, 5, result_path)

    if not os.path.exists(result_path): os.mkdir(result_path)
    # 将平均结果存入excel表格
    average_result_path = result_path + "/CVResult_mean.xlsx"
    with pd.ExcelWriter(average_result_path,
                    engine="openpyxl",
                    ) as writer:  
        result_df.to_excel(writer)
    
    # 将详细结果存入excel表格
    detail_result_path = result_path + "/detail"
    if not os.path.exists(detail_result_path): os.mkdir(detail_result_path)
    for key, value in report_detail_dict.items():
        result_path = os.path.join(detail_result_path, key+'_detail.csv')
        for single_report in value:
            pd.DataFrame(single_report).T.to_csv(result_path, mode='a+')