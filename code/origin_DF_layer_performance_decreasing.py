'''
Description: 验证深度森林模型性能逐层降低。
    需要切换到base环境运行
Author: tanqiong
Date: 2023-10-07 20:20:02
LastEditTime: 2023-10-07 21:36:34
LastEditors: tanqiong
'''

# 完整数据集的baseline
import sys 
import os
import joblib
import copy
import joblib
from deepforest import CascadeForestClassifier
from scores import accuracy, aupr, auroc
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
from preprocessing_data import load_toxric_combination_data, load_toxric

def get_baseline(x, y, cv_list, result_path):
    random_state = 6666
    model_list = [
        CascadeForestClassifier(n_trees=500), 
    ]

    model_name_list = [item.__class__.__name__ for item in model_list]
    metric_name_list = ["AUROC", "AUPR", 'F1', 'ACC', 'Precision', 'Recall']
    result_detail_dict = {}
    result_df = pd.DataFrame(columns=model_name_list, index=metric_name_list)

    random = [i for i in range(20,20+len(cv_list))]
    for model in model_list:
        cv_score_list = []
        predict_dir = os.path.join(result_path, model.__class__.__name__)
        if not os.path.exists(predict_dir): 
            os.mkdir(predict_dir)
        model_not_fit = copy.deepcopy(model)
        for i, (train_idx, test_idx) in enumerate(cv_list):
            x_train, y_train = x[train_idx], y[train_idx]
            x_test, y_test = x[test_idx], y[test_idx]
            model.fit(x_train, y_train)
            # y_pred = model.predict(x_test)
            record_file = os.path.join(predict_dir, f'cv_{i}_eval_test_set_score.txt')
            y_pred_proba = model.predict_proba(
                    X = x_test, 
                    y_test = y_test,
                    record_evaluating=True,
                    evaluate_func_list=[accuracy, aupr, auroc],
                    record_file=record_file,
                )
            y_pred = np.argmax(y_pred_proba, axis=1)

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
    path_prefix = "/data/tq/RECOMB2024/result/exp_origin_DF_performance_decreasing"
    if not os.path.exists(path_prefix): os.mkdir(path_prefix)
    path_prefix = os.path.join(path_prefix, "sklearn_v122")
    if not os.path.exists(path_prefix): os.mkdir(path_prefix)

    # 加载字典数据-0606
    features_dict, labels_df = load_toxric()

    # 加载cv索引
    train_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", dtype=np.int64)
    test_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", dtype=np.int64)
    cv = [(t, v) for t, v in zip(train_ids, test_ids)]

    for n_views in range(4, len(features_dict)+1):
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