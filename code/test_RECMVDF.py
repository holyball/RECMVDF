'''
Description: 在单折数据上训练一个MVUGCForest
Author: tanqiong
Date: 2023-05-14 11:03:42
LastEditTime: 2023-10-09 18:24:13
LastEditors: tanqiong
'''
from MVRECDF.MVRECDF_kernel import MVRECForestClassifier
from MVRECDF.evaluation import accuracy,f1_binary,f1_macro,f1_micro, mse_loss, aupr, auroc
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedKFold,RepeatedStratifiedKFold
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sys
import time

import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)

import os

import shutil
from preprocessing_data import load_data_to_df, load_combine_data_to_df, \
                               load_multiview_data, split_multiview_data, \
                               load_simulation_multiview_data, make_noise_data
from util import reject, save_opinions, get_logger, uncertainty_acc_curve, get_stage_matrix

# import sys
# sys.path.append(os.path.dirname(__file__))

if __name__=="__main__":
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
        features_dict = {k:v.values for k,v in features_dict.items()}
        origin_labels = origin_labels.values
    
    # # 导入模拟小数据集
    # features_dict, origin_labels  = load_simulation_multiview_data(42)
    
    x_train, x_test, y_train, y_test = split_multiview_data(features_dict, origin_labels)
    x_train_list = [x for x in x_train.values()]
    x_test_list = [x for x in x_test.values()]

    # # 删除之前的日志记录
    folder_path = "MVRECDF_info"  # 指定要删除的文件夹路径
    try:
        shutil.rmtree(folder_path)
        print(f"日志文件夹删除成功: {folder_path}")
    except OSError as e:
        print(f"文件夹删除失败: {e}")
    
    model=MVRECForestClassifier(
        n_view=4, n_tree=10, n_fold=5,
        is_stacking_for_boost_features=False,
        is_resample=False,
        cluster_samples_layer=False,
    )
    model.fit(x_train_list, y_train, evaluate_set="all")

    y_proba = model.predict_proba(x_test_list, y_test)
    y_test_pred = np.argmax(y_proba,axis=1)
    print(f"\ntest acc: {accuracy(y_test, y_proba)}\ttest f1: {f1_macro(y_test, y_proba)}")