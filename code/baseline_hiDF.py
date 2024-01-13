'''
Description: hiDF在毒性数据集上的baseline
Author: tanqiong
Date: 2023-05-17 22:46:59
LastEditTime: 2023-05-18 11:49:31
LastEditors: tanqiong
'''
import pandas as pd
import numpy as np
from hiDF.hiDF_utils_threshold import  gcForest_hi
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scores import scores_clf
from preprocessing_data import load_toxric_combination_data
import joblib
import os
# dataset = load_iris()
# features = dataset["data"]
# labels = dataset["target"]
# x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,random_state=666)

num_estimator = 100
num_forests = 5

def get_baseline(x, y, cv_list, result_path):
    random_state = 6666
    model_name = gcForest_hi.__name__
    metric_name_list = ["AUROC", "AUPR", 'F1', 'ACC', 'Precision', 'Recall']
    result_detail_dict = {}
    result_df = pd.DataFrame(columns=[model_name], index=metric_name_list)

    random = [i for i in range(20,20+len(cv_list))]
    cv_score_list = []

    for i, (train_idx, test_idx) in enumerate(cv_list):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        model = gcForest_hi(num_estimator=num_estimator, 
                num_forests=num_forests, 
                num_classes=len(np.unique(y_train)), 
                max_layer=10, 
                max_depth=None, 
                n_fold=5, 
                min_samples_leaf=1)
        acc_list, y_test_pred_proba = model.train(x_train, y_train, x_test, y_test, return_test_proba=True)
        y_pred = np.argmax(y_test_pred_proba, axis=1)
        y_pred_proba = y_test_pred_proba
            
        predict_dir = os.path.join(result_path, model_name)
        if not os.path.exists(predict_dir): 
            os.mkdir(predict_dir)
        pd.DataFrame(
            np.hstack([y_test.reshape((-1, 1)), y_pred.reshape((-1, 1)), y_pred_proba]),
            columns=['y_true', 'y_pred','class_0', 'class_1'],
        ).to_csv(os.path.join(predict_dir, f"cv_{i}_{model_name}.csv"))

        scores = scores_clf(y_test, y_pred_proba).values
        cv_score_list.append(scores)
        del model

    result_detail_dict[model_name] = cv_score_list
    result_df[model_name] = np.mean(cv_score_list, axis=0).reshape(-1)
    result_df[model_name+"_std"] = np.std(cv_score_list, axis=0).reshape(-1)
    return result_detail_dict, result_df


if __name__ == "__main__":
    features_df, labels_df = load_toxric_combination_data()
    feature = features_df.values
    labels = labels_df.values

    ## get baseline ##
    print('get hiDF baseline')
    with open("/data/tq/dataset/toxric/dataset_0403/cache/toxric_cv_0404.pkl", "rb") as handle:
        cv = joblib.load(handle)
    
    x = feature
    y = labels.reshape(-1)
    result_path = "../result/baseline_hiDF_early_stop"
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
