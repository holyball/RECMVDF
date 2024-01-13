'''
Description: 收集特征重要性
Author: tanqiong
Date: 2023-09-07 20:20:17
LastEditTime: 2023-10-20 19:05:54
LastEditors: tanqiong
'''

from MVUGCForest.MVUGCForest import MVUGCForest
from MVUGCForest.evaluation import accuracy,f1_binary,f1_macro,f1_micro, mse_loss, aupr, auroc
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
def get_config():
    config={}
    
    # boost_features
    config["is_stacking_for_boost_features"] = False
    config["boost_feature_type"] = 'probability'            # 'opinion', 'probability', None
   
    # cluster samples
    config["cluster_samples_layer"] = True   
    config["span"]=2
    config["cluster_method"] = "span"                     # 'uncertainty', 'span', 'uncertainty_bins'
    config["n_bins"] = 30
    config["ta"] = 0.7
    
    # resample
    config["is_resample"] = True              # True/Fasle, conflict with 'is_resample'
    config["onn"] = 3                       # outlier nearest neighbour
    config["layer_resample_method"] = "integration"    # "integration", "respective"
    config["accumulation"] = False

    # training 
    config["max_layers"]=10
    config["early_stop_rounds"]=2   
    config["is_weight_bootstrap"] = True
    config["train_evaluation"] = aupr    # accuracy, f1_macro, aupr, auroc, mse_loss
    config["view_opinion_generation_method"] = "mean"  # 'mean'/'joint'/'sum' (sum==mean)
    config["is_save_model"] = False
    # config["random_state"]=666     # 669172976

    # prediction
    config["use_layerwise_opinion"] = True              # 是否考虑层间联合opinion
    config["bypass_prediction"] = True

    # node configs
    config["uncertainty_basis"] = "no_uncertainty"    # "entropy"/"evidence"
    config["evidence_type"] = "probability"     # "knn"(废弃) / "probability" 
    config["estimator_configs"]=[]
    for _ in range(2):
        config["estimator_configs"].append({"n_fold":3, 
                                            "type": "RandomForestClassifier",
                                            ### sklearn parameters ###
                                            "n_estimators": 50, 
                                            # "max_features": None,   # None/'sqrt'/float
                                            # "max_samples": 0.7,
                                            "max_depth": None, 
                                            "n_jobs": -1, 
                                            "min_samples_leaf": 1,
                                            # "min_weight_fraction_leaf": 2/504,
                                            "max_features": None,
                                            # "bootstrap": False,
                                            })
    for _ in range(2):
        config["estimator_configs"].append({"n_fold": 3, 
                                            "type": "ExtraTreesClassifier",
                                            ### sklearn parameters ###
                                            "n_estimators": 50, 
                                            # "max_features": None,  # None/'sqrt'/float
                                            # "max_samples": 0.7, 
                                            "max_depth": None, 
                                            "n_jobs": -1, 
                                            "min_samples_leaf": 1, 
                                            # "min_weight_fraction_leaf": 2/504,
                                            "max_features": None,
                                            # "bootstrap": False,
                                            })
    return config
if __name__=="__main__":
    from sklearn.datasets import load_digits, load_breast_cancer
    # 导入multiview数据集
    # features_dict, origin_labels = load_multiview_data('all')
    # features_dict = {i:features.values for i, features in enumerate(features_dict.values())}
    # origin_labels = origin_labels.values

    # 导入缓存的数据集
    import joblib
    with open("/data/tq/dataset/toxric/dataset_0403/cache/features_dict_labels_0613.pkl", "rb") as handle:
        features_dict, origin_labels = joblib.load(handle)

    # # 导入模拟小数据集
    # features_dict, origin_labels  = load_simulation_multiview_data(42)
    
    x_train, x_test, y_train, y_test = split_multiview_data(features_dict, origin_labels)
    print(len(y_test))
    config=get_config()
    config['logger_path'] = "MVUGCForest_info_feature_importance"

    # # 删除之前的日志记录
    folder_path = config['logger_path']  # 指定要删除的文件夹路径
    try:
        shutil.rmtree(folder_path)
        print(f"日志文件夹删除成功: {folder_path}")
    except OSError as e:
        print(f"文件夹删除失败: {e}")
    
    gc=MVUGCForest(config)
    gc.fit_multiviews(x_train,y_train)

    """收集特征重要性"""
    for v, view in gc.views.items():
        excel_handle = pd.ExcelWriter(f"/data/tq/RECOMB2024/result/特征重要性/view_{v}.xlsx")
        for l, layer in enumerate(view.layers):
            feature_names = layer.feature_names_
            feature_importances = layer.feature_importances_
            df = pd.DataFrame(np.reshape(feature_importances, (-1, 1)), index=feature_names)
            df.to_excel(excel_handle, sheet_name=f"layer_{l}")
        excel_handle.close()

            
    