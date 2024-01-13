'''
Description: MVUGCForest的交叉验证版本, 包含了所有view组合的结果
             所有view 的组合, 以及某个组合下的所有组合 结果都存下来了!
Author: tanqiong
Date: 2023-05-29 20:36:55
LastEditTime: 2023-10-12 21:25:15
LastEditors: tanqiong
'''

# 完整数据集上(dataset_0403)的实验
from MVUGCForest.MVUGCForest import MVUGCForest, joint_multi_opinion
from MVUGCForest.evaluation import accuracy,f1_binary,f1_macro,f1_micro, mse_loss, aupr, auroc
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
import shutil

from itertools import combinations

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
                               split_multiview_data, split_multiview_data_cv, load_simulation_multiview_data_cv
from util import reject, save_opinions, get_logger

def opinion_to_proba(opinion: np.ndarray)->np.ndarray:
    """将opinion转为为概率
    Parameters
    ----------
    opinion: ndarray, 有可能会存在空洞的观点!
    """
    proba = opinion[:, :-1].copy()
    proba /= np.sum(proba, axis=1)[:, np.newaxis]
    if np.sum(np.isnan(proba)) != 0:
        np.nan_to_num(proba, 0)
    return proba

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
                                            "n_estimators": 5, 
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
                                            "n_estimators": 5, 
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
    
    # # 导入multiview数据集
    # features_dict, labels = load_multiview_data(type='all')
    # features_dict = {i:features.values for i, features in enumerate(features_dict.values())}
    # labels = labels.values

    # 导入缓存数据
    with open("/data/tq/dataset/toxric/dataset_0403/cache/features_dict_labels_0613.pkl", "rb") as handle:
        features_dict, labels = joblib.load(handle)
    train_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", dtype=np.int64)
    test_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", dtype=np.int64)
    cv = [(t, v) for t, v in zip(train_ids, test_ids)]

    # # 模拟数据集
    # features_dict, labels, cv  = load_simulation_multiview_data_cv(42)
    # cv = cv[:2]
        
    n_fold = len(cv)
    
    path_prefix = args.result_path if args.result_path is not None else "/data/tq/RECOMB2024/result/enchanced_cascade_1012"
    if not os.path.exists(path_prefix):
        os.mkdir(path_prefix)

    for n_views in range(2, len(features_dict)+1):
        for c in combinations(range(len(features_dict)), n_views):

            # 视图特征组合
            str_name = str(c).replace(',', '_')
            result_path = path_prefix+f"/{str_name}"
            if not os.path.exists(result_path):
                os.mkdir(result_path)
            else:
                continue
            # 打印结果存储的文件夹
            print(result_path)

            # 视图特征组合, 生成包含特定组合的字典
            features_train_dict = {ni:features_dict[ni] for ni in c}

            # 收集交叉验证结果的容器
            cv_score_list_sum_view = []     # shape of (#folds, )
            cv_score_list_single_view = []  # shape of (#folds, #views, #metrics, 1)
            cv_score_list_comb_view = []    # shape of (#folds, #combs, #metrics, 1)
            # 开始25折交叉验证
            for i, (x_train, x_test, y_train, y_test) in enumerate(split_multiview_data_cv(features_train_dict, labels, cv)):

                y_train = y_train.squeeze()
                y_test = y_test.squeeze()

                config=get_config()
                # 增加logger路径
                fold_dir = os.path.join(result_path, f"fold_{i}")
                if not os.path.exists(fold_dir): os.mkdir(fold_dir)

                # 每折跑五次, 然后取最好
                acc_best = 0
                for finding in range(1):
                    
                    # 增加logger路径
                    config["logger_path"] = os.path.join(fold_dir, f"finding_{finding}")
                    if not os.path.exists(config["logger_path"]): os.mkdir(config["logger_path"])

                    model=MVUGCForest(config)
                    model.fit_multiviews(x_train, y_train)
                    # y_pred_proba = model.predict_proba(x_test, y_test)
                    y_pred_proba_cur, y_pred_opinion_cur = model.predict_opinion(x_test, y_test)

                    y_pred = np.argmax(y_pred_proba_cur, axis=1).squeeze()
                    
                    # 检查是否获得更好的评估分数
                    eval_func = config["train_evaluation"]
                    acc_cur = eval_func(y_test, y_pred_proba_cur)
                    if (acc_cur>acc_best):
                        acc_best = acc_cur
                        y_pred_proba = y_pred_proba_cur
                        y_pred_opinion = y_pred_opinion_cur
                        best_finding = finding  # 最佳的轮次
                        best_finding_dir = config["logger_path"]
                    
                # 重命名文件夹
                new_best_finding_dir_name = os.path.join(os.path.dirname(best_finding_dir), "best")
                shutil.move(best_finding_dir, new_best_finding_dir_name)

                # 记录结果
                model_name = model.__class__.__name__
                predict_dir = os.path.join(result_path, model_name)
                if not os.path.exists(predict_dir): 
                    os.mkdir(predict_dir)
                pd.DataFrame(
                    np.hstack([y_test.reshape((-1, 1)), y_pred.reshape((-1, 1)), y_pred_proba, y_pred_opinion[:, -1].reshape((-1, 1))]),
                    columns=['y_true', 'y_pred','class_0', 'class_1', 'uncertainty'],
                ).to_csv(os.path.join(predict_dir, f"cv_{i}_{model_name}.csv"))

                # 记录综合view(所有)的结果
                scores = scores_clf(y_test, y_pred_proba).values
                cv_score_list_sum_view.append(scores)
                
                # 记录每个view的结果
                scores_views = []   # shape of (#views, )
                layer_view_opinion_dict_predict = model.layer_view_opinion_dict_predict
                depth = model.best_layer_id
                for opinion_list in layer_view_opinion_dict_predict.values():
                    opinion_final_layer = opinion_list[-1]
                    y_pred_proba_view = opinion_to_proba(opinion_final_layer)
                    score_view = scores_clf(y_test, y_pred_proba_view).values
                    scores_views.append(score_view)
                cv_score_list_single_view.append(scores_views)

                # 记录组合view的结果
                scores_views_comb = []  # shape of (#combs, )
                for _n_v in range(2, n_views):
                    for _c_v in combinations(c, _n_v):

                        views_idx = [c.index(feature_view_id) for feature_view_id in _c_v]
                        opinion_list = [layer_view_opinion_dict_predict[f"view_{v}"][-1] for v in views_idx]     
                        opinion_comb = joint_multi_opinion(opinion_list)
                        y_pred_proba_view_comb = opinion_to_proba(opinion_comb)
                        score_views_comb = scores_clf(y_test, y_pred_proba_view_comb).values
                        scores_views_comb.append(score_views_comb)
                cv_score_list_comb_view.append(scores_views_comb)

                # if i < n_fold-1:
                #     del model
                time.sleep(10)

            # 记录综合view得分的均值,方差
            result_df = pd.DataFrame(index=metric_name_list)
            result_df[model_name] = np.mean(cv_score_list_sum_view, axis=0).reshape(-1)
            result_df[model_name+"_std"] = np.std(cv_score_list_sum_view, axis=0).reshape(-1)
            
            # 记录每个view(单个)
            cv_score_list_single_view = np.squeeze(cv_score_list_single_view)
            for vi in range(len(cv_score_list_single_view[0])):
                result_df[model_name+f'_view_{vi}'] = np.mean(cv_score_list_single_view[:,vi,:], axis=0).reshape(-1)
            for vi in range(len(cv_score_list_single_view[0])):
                result_df[model_name+f"_std_view_{vi}"] = np.std(cv_score_list_single_view[:,vi,:], axis=0).reshape(-1)
            
            # 记录每个组合view(除了全部view与单个view的结果)
            cv_score_list_comb_view = np.squeeze(cv_score_list_comb_view)
            vi = 0
            for _n_v in range(2, n_views):
                for _c_v in combinations(c, _n_v):
                    views_comb_name = str(_c_v).replace(',', '_')
                    result_df[model_name+f'_view_{views_comb_name}'] = np.mean(cv_score_list_comb_view[:,vi,:], axis=0).reshape(-1)
                    vi += 1
            vi = 0
            for _n_v in range(2, n_views):
                for _c_v in combinations(c, _n_v):
                    views_comb_name = str(_c_v).replace(',', '_')
                    result_df[model_name+f'_std_view_{views_comb_name}'] = np.std(cv_score_list_comb_view[:,vi,:], axis=0).reshape(-1)
                    vi += 1

            columns = result_df.columns.to_list()
            contains_std_list = [element for element in columns if 'std' in element]
            not_contains_std_list = [element for element in columns if 'std' not in element]
            columns = not_contains_std_list + contains_std_list
            result_df = result_df[columns]

            result_df.to_csv(os.path.join(result_path, f"MVUGCForest_{result_path.split('/')[-1]}_cv.csv"))


