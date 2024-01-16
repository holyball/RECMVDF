'''
Description: 在单折数据上训练一个MVUGCForest
Author: tanqiong
Date: 2023-05-14 11:03:42
LastEditTime: 2023-10-09 18:24:13
LastEditors: tanqiong
'''
from util import reject, save_opinions, get_logger, uncertainty_acc_curve, get_stage_matrix
from preprocessing_data import load_data_to_df, load_combine_data_to_df, \
    load_multiview_data, split_multiview_data, \
    load_simulation_multiview_data, make_noise_data
import shutil
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedKFold, RepeatedStratifiedKFold)

from deepforest import CascadeForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)

import sys
import time

import warnings
from MVRECDF.evaluation import (accuracy, f1_binary, f1_macro, f1_micro,
                                mse_loss, aupr, auroc)
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)


# import sys
# sys.path.append(os.path.dirname(__file__))

if __name__ == "__main__":
    from sklearn.datasets import load_digits, load_breast_cancer
    # # 完整地导入multiview数据集, 速度慢
    # features_dict, origin_labels = load_multiview_data('all')
    # features_dict = {i:features.values for i, features in enumerate(features_dict.values())}
    # origin_labels = origin_labels.values
    # features_dict, origin_labels = load_multiview_data('all')

    # # 从缓存文件中导入数据集, 速度快
    import joblib
    with open("/data/tq/dataset/toxric/dataset_0403/cache/features_dict_labels_0613_v122.pkl", "rb") as handle:
        features_dict, origin_labels = joblib.load(handle)
        features_dict = {k: v.values for k, v in features_dict.items()}
        origin_labels = origin_labels.values

    # # 导入模拟小数据集
    # features_dict, origin_labels  = load_simulation_multiview_data(42)

    x_train, x_test, y_train, y_test = split_multiview_data(
        features_dict, origin_labels)
    x_train_list = [x for x in x_train.values()]
    x_test_list = [x for x in x_test.values()]
    
    x_train = np.hstack(x_train_list[2:4])
    x_test = np.hstack(x_test_list[2:4])

    random_state = 42

    model_list = [
        RandomForestClassifier(n_estimators=500), 
        ExtraTreesClassifier(n_estimators=500),
        XGBClassifier(n_estimators=500, eval_metric=['logloss','auc','error']), 
        CatBoostClassifier(n_estimators=500),
        LGBMClassifier(n_estimators=500),
        CascadeForestClassifier(n_trees=500), 
        GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500),
        AdaBoostClassifier(n_estimators=500,),
    ]
    for model in model_list:
        model.fit(x_train, y_train)

        y_proba = model.predict_proba(x_test)
        # y_test_pred = np.argmax(y_proba, axis=1)

        with open('test_baseline_view34.txt', 'a') as file:  # 使用 'a' 模式以追加的方式写入文件
            file.write(f"model: {model.__class__.__name__}\n")
            file.write(
                f"test acc: {accuracy(y_test, y_proba)}\ttest f1: {f1_macro(y_test, y_proba)}\n\n")