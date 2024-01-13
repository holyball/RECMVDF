'''
Description: 毒性数据集baseline, 将多个views的数据分开训练，最后平均所有view的学习器的输出 
             需要切换到base环境, 
             0529: 得到所有模态组合的baseline, 并保存实验结果
Author: tanqiong
Date: 2023-04-04 16:12:13
LastEditTime: 2023-06-14 12:15:36
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

from itertools import combinations

import copy
import numpy as np
import pandas as pd
from scores import scores_clf
from preprocessing_data import load_toxric_combination_data, split_multiview_data_cv, load_simulation_multiview_data_cv, load_multiview_data

def get_baseline(x_dict, y, cv_list, result_path):
    random_state = 6666
    model_list = [
        # AdaBoostClassifier(n_estimators=500,),
        # ExtraTreesClassifier(n_estimators=500,
        #                         criterion="gini",
        #                         max_depth=None,
        #                         min_samples_split=2,
        #                         min_samples_leaf=1,
        #                         min_weight_fraction_leaf=0.,
        #                         max_features="auto",
        #                         max_leaf_nodes=None,
        #                         min_impurity_decrease=0.,
        #                         min_impurity_split=None,
        #                         bootstrap=False,
        #                         oob_score=False,
        #                         n_jobs=-1,
        #                         random_state=random_state,
        #                         verbose=0,
        #                         warm_start=False,
        #                         class_weight=None,
        #                         ccp_alpha=0.0,
        #                         max_samples=None),
        # RandomForestClassifier(n_estimators=500,
        #                         criterion="gini",
        #                         max_depth=None,
        #                         min_samples_split=2,
        #                         min_samples_leaf=1,
        #                         min_weight_fraction_leaf=0.,
        #                         max_features="auto",
        #                         max_leaf_nodes=None,
        #                         min_impurity_decrease=0.,
        #                         min_impurity_split=None,
        #                         bootstrap=False,
        #                         oob_score=False,
        #                         n_jobs=None,
        #                         random_state=random_state,
        #                         verbose=0,
        #                         warm_start=False,
        #                         class_weight=None,
        #                         ccp_alpha=0.0,
        #                         max_samples=None), 
        # GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500,
        #          subsample=1.0, criterion='friedman_mse', min_samples_split=2,
        #          min_samples_leaf=1, min_weight_fraction_leaf=0.,
        #          max_depth=10, min_impurity_decrease=0.,
        #          min_impurity_split=None, init=None,
        #          random_state=None, max_features=None, verbose=0,
        #          max_leaf_nodes=None, warm_start=False,
        #          presort='deprecated', validation_fraction=0.1,
        #          n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0), 
        CascadeForestClassifier(), 
        XGBClassifier(n_estimators=500, eval_metric=['logloss','auc','error']), 
        CatBoostClassifier(n_estimators=500),
        LGBMClassifier(n_estimators=500),
        # MLPClassifier(hidden_layer_sizes=(62,154,274,343,258,)),
    ]

    model_name_list = [item.__class__.__name__ for item in model_list]
    metric_name_list = ["AUROC", "AUPR", 'F1', 'ACC', 'Precision', 'Recall']
    result_detail_dict = {}
    result_df = pd.DataFrame(columns=model_name_list, index=metric_name_list)

    random = [i for i in range(20,20+len(cv_list))]
    for model_not_fit in model_list:
        model_name = model_not_fit.__class__.__name__
        cv_score_list_sum_view = []
        cv_score_list_single_view = []
        # model_not_fit = copy.deepcopy(model)

        for i, (x_train_dict, x_test_dict, y_train, y_test) in enumerate(split_multiview_data_cv(x_dict, y, cv_list)):
            scores_view = []
       
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()
            y_test_pred_proba = 0
            for x_train,x_test in zip(x_train_dict.values(), x_test_dict.values()):
                if model_name in ['LogisticRegression', 'RidgeClassifier', 'LinearSVC']:
                    ss = StandardScaler()
                    x_train = ss.fit_transform(x_train)
                    x_test = ss.transform(x_test)
                model = copy.deepcopy(model_not_fit)
                model.fit(x_train, y_train)
                # y_pred = model.predict(x_test)
                y_pred_proba_view = model.predict_proba(x_test)
                y_test_pred_proba += y_pred_proba_view

                y_pred_view = np.argmax(y_pred_proba_view, axis=1)
                scores_view.append(scores_clf(y_test, y_pred_proba_view))

            y_test_pred_proba /= len(x_train_dict)
            y_test_pred = np.argmax(y_test_pred_proba, axis=1)

            predict_dir = os.path.join(result_path, model_name)
            if not os.path.exists(predict_dir): 
                os.mkdir(predict_dir)
            pd.DataFrame(
                np.hstack([y_test.reshape((-1, 1)), y_test_pred.reshape((-1, 1)), y_test_pred_proba]),
                columns=['y_true', 'y_pred','class_0', 'class_1'],
            ).to_csv(os.path.join(predict_dir, f"cv_{i}_{model_name}.csv"))

            scores_sum = scores_clf(y_test, y_test_pred_proba).values
            cv_score_list_sum_view.append(scores_sum)
            cv_score_list_single_view.append(scores_view)
            del model
            # model = copy.deepcopy(model_not_fit)

        result_detail_dict[model_name] = cv_score_list_sum_view
        result_df[model_name] = np.mean(cv_score_list_sum_view, axis=0).reshape(-1)
        result_df[model_name+"_std"] = np.std(cv_score_list_sum_view, axis=0).reshape(-1)
        cv_score_list_single_view = np.array(cv_score_list_single_view).squeeze()     # shape of (#folds, #views, #metrics)
        for vi in range(len(cv_score_list_single_view[0])):
            result_df[model_name+f'_view_{vi}'] = np.mean(cv_score_list_single_view[:,vi,:], axis=0).reshape(-1)
        for vi in range(len(cv_score_list_single_view[0])):
            result_df[model_name+f"_std_view_{vi}"] = np.std(cv_score_list_single_view[:,vi,:], axis=0).reshape(-1)

        
    columns = result_df.columns.to_list()
    contains_std_list = [element for element in columns if 'std' in element]
    not_contains_std_list = [element for element in columns if 'std' not in element]
    columns = not_contains_std_list + contains_std_list
    result_df = result_df[columns]
        
    return result_detail_dict, result_df
    
if __name__ == "__main__":
    # 加载数据
    print('get baseline apart')
    # # 模拟数据
    # features_dict, labels, cv = load_simulation_multiview_data_cv(42)

    # 药物毒性数据, 适用于所有
    features_dict, origin_labels = load_multiview_data('all')
    features_dict = {i:features.values for i, features in enumerate(features_dict.values())}
    labels = origin_labels.values
    # cv, 适用于所有
    train_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", dtype=np.int64)
    test_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", dtype=np.int64)
    cv = [(t, v) for t, v in zip(train_ids, test_ids)]

    # # 导入模拟小数据集
    # features_dict, labels, cv  = load_simulation_multiview_data_cv(42)

    result_path_prefix = "../result/baseline_result_apart_0613"
    if not os.path.exists(result_path_prefix): os.mkdir(result_path_prefix)

    # 组合所有模态
    for n_views in range(2,len(features_dict)+1):
    # for n_views in range(4,len(features_dict)+1):
        for c in combinations(range(len(features_dict)), n_views):
            str_name = str(c).replace(',', '_')
            result_path = result_path_prefix + f"/v122_{str_name}"
            if not os.path.exists(result_path): os.mkdir(result_path)
            features_train_dict = {ni:features_dict[ni] for ni in c} 
            report_detail_dict, result_df = get_baseline(features_train_dict, labels, cv, result_path)

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