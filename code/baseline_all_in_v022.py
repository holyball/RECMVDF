'''
Description: 毒性数据集baseline,
             特征组合方式: all in
             sklearn-v022, 需要切换到env_hiDF环境
             0613 更新数据集
Author: tanqiong
Date: 2023-04-04 16:12:13
LastEditTime: 2023-06-13 14:29:49
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
# from deepforest import CascadeForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import joblib

import copy
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

from scores import scores_clf
from preprocessing_data import load_toxric_combination_data, load_toxric


def get_baseline(x, y, cv_list, result_path):
    random_state = 6666
    model_list = [
        AdaBoostClassifier(n_estimators=500,),
        ExtraTreesClassifier(n_estimators=500,
                                criterion="gini",
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.,
                                max_features="auto",
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.,
                                min_impurity_split=None,
                                bootstrap=False,
                                oob_score=False,
                                n_jobs=-1,
                                random_state=random_state,
                                verbose=0,
                                warm_start=False,
                                class_weight=None,
                                ccp_alpha=0.0,
                                max_samples=None),
        RandomForestClassifier(n_estimators=500,
                                criterion="gini",
                                max_depth=None,
                                min_samples_split=2,
                                min_samples_leaf=1,
                                min_weight_fraction_leaf=0.,
                                max_features="auto",
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.,
                                min_impurity_split=None,
                                bootstrap=False,
                                oob_score=False,
                                n_jobs=None,
                                random_state=random_state,
                                verbose=0,
                                warm_start=False,
                                class_weight=None,
                                ccp_alpha=0.0,
                                max_samples=None), 
        GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=10, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 presort='deprecated', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0), 
        # CascadeForestClassifier(n_trees=500), 
        # XGBClassifier(n_estimators=500, eval_metric=['logloss','auc','error']), 
        # CatBoostClassifier(n_estimators=500),
        # LGBMClassifier(n_estimators=500),
        # MLPClassifier(hidden_layer_sizes=(62,154,274,343,258,)),
    ]

    model_name_list = [item.__class__.__name__ for item in model_list]
    metric_name_list = ["AUROC", "AUPR", 'F1', 'ACC', 'Precision', 'Recall']
    result_detail_dict = {}
    result_df = pd.DataFrame(columns=model_name_list, index=metric_name_list)

    random = [i for i in range(20,20+len(cv_list))]
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
    
if __name__ == "__main__":

    # 路径前缀
    path_prefix = "../result/baseline_result_all_in_0613"
    if not os.path.exists(path_prefix): os.mkdir(path_prefix)
    path_prefix = os.path.join(path_prefix, "sklearn_v022")
    if not os.path.exists(path_prefix): os.mkdir(path_prefix)

    # 加载字典数据-0606
    features_dict, labels_df = load_toxric()

    # 加载cv索引
    train_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", dtype=np.int64)
    test_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", dtype=np.int64)
    cv = [(t, v) for t, v in zip(train_ids, test_ids)]
    
    for n_views in range(2, len(features_dict)+1):
        for c in combinations(range(len(features_dict)), n_views):
            # 创建建存储各个组合结果的文件夹
            str_name = str(c).replace(',', '_')
            result_path = path_prefix + f"/{path_prefix.split('sklearn_')[-1]}_{str_name}"
            if not os.path.exists(result_path): os.mkdir(result_path)

            features_train_dict = {ni:features_dict[ni] for ni in c}
            features_df = pd.concat(features_train_dict, axis=1)
            features = features_df.values
            labels = labels_df.values 
            
            ## get baseline ##
            print('get baseline')            
            
            x = features
            y = labels.reshape(-1)
            
            if not os.path.exists(result_path): os.mkdir(result_path)
            report_detail_dict, result_df = get_baseline(x, y, cv, result_path)

            if not os.path.exists(result_path): os.mkdir(result_path)
            # 将平均结果存入excel表格
            average_result_path = result_path + f"/CVResult_mean_{result_path.split('/')[-1]}.xlsx"
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