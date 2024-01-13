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
from preprocessing_data import load_toxric_combination_data

def get_baseline(x, y, cv_list, result_path):
    model_list = [
        AdaBoostClassifier(),
        ExtraTreesClassifier(n_jobs=-1), 
        RandomForestClassifier(n_jobs=-1), 
        GradientBoostingClassifier(random_state=66666), 
        CascadeForestClassifier(), 
        XGBClassifier(eval_metric=['logloss','auc','error']), 
        CatBoostClassifier(),
        LGBMClassifier(),
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
    
if __name__ == "__main__":
    # 加载数据
    # features_df, labels_df = load_combine_data_to_df()
    features_df, labels_df = load_toxric_combination_data()
    feature = features_df.values
    label = labels_df.values

    ## get baseline ##
    print('get baseline')

    rsfk = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=666)
    cv = [(train_idx, test_idx) for train_idx, test_idx in rsfk.split(feature, label)]
    with open("/data/tq/dataset/toxric/dataset_0403/cache/toxric_cv_0404.pkl", "wb") as handle:
        joblib.dump(cv, handle)
    
    x = feature
    y = label.reshape(-1)
    result_path = "../result/baseline_result"
    report_detail_dict, result_df = get_baseline(x, y, cv, result_path)

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