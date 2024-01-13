'''
Description: MVUGCForest的交叉验证, 可以选择使用五折交叉验证, 缩短时间
Author: tanqiong
Date: 2023-05-13 23:20:33
LastEditTime: 2023-06-13 22:54:29
LastEditors: tanqiong
'''
# 完整数据集上(dataset_0403)的实验
from MVUGCForest.MVUGCForest import MVUGCForest
from MVUGCForest.evaluation import accuracy,f1_binary,f1_macro,f1_micro, mse_loss
from sklearn.model_selection import train_test_split,StratifiedKFold,RepeatedKFold,RepeatedStratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score
import warnings
import joblib
import pandas as pd
from scores import scores_clf, metric_name_list
import argparse
import sys
import time

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=4)

# 创建解析器
parser = argparse.ArgumentParser(description='cross validation for MVUGForest')
# 添加参数
parser.add_argument('-rp', type=str, dest='result_path', 
                    help='dictionary for saving results')
# 解析参数
args = parser.parse_args()

# 导包: 数据, 日志
import os
from preprocessing_data import load_data_to_df, load_multiview_data, \
                               split_multiview_data, split_multiview_data_cv
from util import reject, save_opinions, get_logger

def get_config():
    config={}
    # config["random_state"]=666     # 669172976
    config["max_layers"]=10
    config["early_stop_rounds"]=3
    config["span"]=2
    config["is_stacking_for_boost_features"] = False
    config["is_RIT"] = True
    config["is_intersection_inner_view"] = True
    config["is_intersection_cross_views"] = True
    config["ts_encoding"] = False                       # TS编码 True('uncertainty'), False,  'opinion', 'uncertainty', 'proba'
    config["is_save_model"] = False 
    config["train_evaluation"]=accuracy                 # accuracy, f1_binary, f1_macro, f1_micro, mse_loss
    config["boost_feature_type"] = 'opinion'            # 'opinion', 'probability', None
    config["view_opinion_generation_method"] = "joint"  # 'mean'/'joint'/'sum'
    config["use_layerwise_opinion"] = False              # 是否考虑层间联合opinion
    # config["is_resample"] = False
    config["group_sample"] = "uncertainty_bins"                     # 'uncertainty', 'span', 'uncertainty_bins'
    config["n_bins"] = 30
    config["ta"] = 0.7
    config["is_resample_in_layer"] = True              # True/Fasle, conflict with 'is_resample'
    config["accumulation"] = True
    config["estimator_configs"]=[]
    for _ in range(2):
        config["estimator_configs"].append({"n_fold":5, 
                                            "type": "RandomForestClassifier",
                                            "uncertainty_basis": "evidence",    # "entropy"/"evidence"
                                            "evidence_type": "probability",     # "knn" / "probability"
                                            "act_func": "approx_step",                 # 'approx_step', 'ReLU', None
                                            "W_type": "sum",               # 'n_class', 'n_tree', 'sum', 'variable' 
                                            "use_kde": False,            
                                                    # 是否使用kde
                                            ### sklearn parameters ###
                                            "n_estimators": 10, 
                                            "max_features": None,   # None/'sqrt'/float
                                            # "max_samples": 0.7,
                                            "max_depth": None, 
                                            "n_jobs": -1, 
                                            "min_samples_leaf": 2,
                                            # "max_features": None,
                                            # "bootstrap": False,
                                            })
    for _ in range(2):
        config["estimator_configs"].append({"n_fold": 5, 
                                            "type": "ExtraTreesClassifier",
                                            "uncertainty_basis": "evidence",    # "entropy", "evidence"
                                            "evidence_type": "probability",     # "knn" , "probability"
                                            "act_func": "approx_step",          # 'approx_step', 'ReLU', None
                                            "W_type": "sum",                    # 'n_class', 'n_tree', 'sum', 'variable' 
                                            "use_kde": False,                    # 是否使用kde

                                            ### sklearn parameters ###
                                            "n_estimators": 10, 
                                            "max_features": None,  # None/'sqrt'/float
                                            # "max_samples": 0.7, 
                                            "max_depth": None, 
                                            "n_jobs": -1, 
                                            "min_samples_leaf": 2, 
                                            # "max_features": None,
                                            # "bootstrap": False,
                                            })

    return config

def get_configs_grid():
    """网格搜索参数列表"""
    from sklearn.model_selection import ParameterGrid

    global_configs = {
        "group_smaple": ['uncertainty_bins'],
        "n_bins": np.arange(10, 121, 10)
    }

    global_grids = list(ParameterGrid(global_configs))
    for glo in global_grids:
        config_new = get_config()   # 基于get_config()函数
        config_new.update(glo)

        yield config_new

def check_params(config:dict):
    """检查参数是否冲突"""
    
    if config.get("is_resample") == True and config["is_intersection_cross_views"] == True:
        return False
    
    elif (config.get("is_resample") == False and config["is_resample_in_layer"] == False) and config["accumulation"] == True:
        return False
    
    elif config.get("is_resample") == True and config["is_resample_in_layer"] == True:
        return False
    
    else: 
        return True
    
def grid_cv_MVGCForest():
    """基于药物毒性数据集做网格搜索主要超参数
    注意修改: result_path, 
    """
    result_path = "/home/tq/uncertainty_estimation_0403/result/grid_search_0612_n_bins_5cv"
    result_father_dir = result_path
    if not os.path.exists(result_path):
        os.mkdir(result_path)


    # 导入缓存数据
    with open("/data/tq/dataset/toxric/dataset_0403/cache/features_dict_labels_0613.pkl", "rb") as handle:
        features_dict, labels = joblib.load(handle)
    train_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", dtype=np.int64)
    test_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", dtype=np.int64)
    cv = [(t, v) for t, v in zip(train_ids, test_ids)]
    cv = cv[:5]
    n_fold = len(cv)

    params_list = list(get_configs_grid())
    print(f"共有{len(params_list)}组参数")

    all_res_df = pd.DataFrame(get_configs_grid())
    # 参数配置_文件夹名称映射
    all_res_df["dir_name"] = [f"config_{ni}" for ni, _ in enumerate(params_list) ]
    # 检查哪些参数是合法的
    for ni, config in enumerate(params_list):
        if check_params(config):
            all_res_df.loc[ni, 'param_is_valid'] = 'valid'
    all_res_df.to_csv(f"{result_father_dir}/grid_search_results_{result_father_dir.split('/')[-1]}.csv")

    # 参数搜索
    for ni, config in enumerate(params_list):
        # if ni < 59: continue
        # 更新result_path, 每种参数对应一个result_path
        result_path = os.path.join(result_father_dir, f"config_{ni}")
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        # 如果参数不冲突
        if all_res_df.loc[ni, 'param_is_valid'] == 'valid':
            
            # 开始五折交叉验证
            cv_score_list = []
            for i, (x_train, x_test, y_train, y_test) in enumerate(split_multiview_data_cv(features_dict, labels, cv)):
        
                y_train = y_train.squeeze()
                y_test = y_test.squeeze()

                # 增加logger路径
                config["logger_path"] = os.path.join(result_path, f"fold_{i}")
                if not os.path.exists(config["logger_path"]): os.mkdir(config["logger_path"])
                
                model = MVUGCForest(config=config)
                model.fit_multiviews(x_train, y_train, True)
                # y_pred = gc.predict(x_test, y_test)
                y_pred_proba, y_pred_opinion = model.predict_opinion(x_test, y_test)

                y_pred = np.argmax(y_pred_proba, axis=1).squeeze()

                predict_dir = os.path.join(result_path, model.__class__.__name__)
                if not os.path.exists(predict_dir): 
                    os.mkdir(predict_dir)
                pd.DataFrame(
                    np.hstack([y_test.reshape((-1, 1)), y_pred.reshape((-1, 1)), y_pred_proba, y_pred_opinion[:, -1].reshape((-1, 1))]),
                    columns=['y_true', 'y_pred','class_0', 'class_1', 'uncertainty'],
                ).to_csv(os.path.join(predict_dir, f"cv_{i}_{model.__class__.__name__}.csv"))
                scores = scores_clf(y_test, y_pred_proba).values
                cv_score_list.append(scores)

                if i<n_fold-1:
                    del model
                time.sleep(10)
            
            result_df = pd.DataFrame(index=metric_name_list)
            result_df[model.__class__.__name__] = np.mean(cv_score_list, axis=0).reshape(-1)
            result_df[model.__class__.__name__+"_std"] = np.std(cv_score_list, axis=0).reshape(-1)
            result_df.to_csv(os.path.join(result_path, f"MVUGCForest_{result_path.split('/')[-1]}_cv.csv"))
            # 把每种参数配置下的结果更新到all_res_df中
            all_res_df.loc[ni, metric_name_list] = result_df[model.__class__.__name__].values   # 均值
            all_res_df.loc[ni, [item+"_std" for item in metric_name_list]] = result_df[model.__class__.__name__+"_std"].values # 标准差
            # 释放资源
            del model
            time.sleep(10)

        # 如果参数之间冲突
        else:
            pass

    # 将all_res_df 按照ACC的大小, 降序排序
    all_res_df.sort_values(by='ACC', ascending=False)
    # 将all_res_df 存储到对应路径下
    all_res_df.to_csv(f"{result_father_dir}/grid_search_results_{result_father_dir.split('/')[-1]}.csv")


if __name__=="__main__":

    # 网格搜索
    grid_cv_MVGCForest()
    