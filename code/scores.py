#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""评分函数    
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
                            recall_score, f1_score, average_precision_score

def accuracy(y_true,y_proba) -> float:
    not_vague_idx = np.sum(y_proba, axis=1)!=0
    y_pred = np.argmax(y_proba, axis=1) 
    return accuracy_score(y_true[not_vague_idx],y_pred[not_vague_idx])

def f1_binary(y_true,y_proba):
    not_vague_idx = np.sum(y_proba, axis=1)!=0
    y_pred = np.argmax(y_proba, axis=1)
    f1=f1_score(y_true[not_vague_idx],y_pred[not_vague_idx],average="binary")
    return f1

def auroc(y_true, y_proba):
        
    # 处理y_true只有一个类别的情况:
    labels = np.unique(y_true)
    if (len(labels)==1):
        if labels[0] == 0:
            y_true = np.append(y_true, 1)
            y_proba = np.vstack( [y_proba, [[0,1]] ] )
        elif labels[0]==1:
            y_true = np.append(y_true, 0)
            y_proba = np.vstack( [y_proba, [[1, 0]] ] )
        else:
            raise ValueError("Only Support Binary Classification now!")
        
    not_vague_idx = np.sum(y_proba, axis=1)!=0

    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, -1]
    try:
        auroc_score = roc_auc_score(y_true[not_vague_idx], y_proba[not_vague_idx], multi_class="ovr")
    except ValueError:
        print(y_proba[not_vague_idx])
    return auroc_score

def aupr(y_true, y_proba):
    # 处理y_true只有一个类别的情况:
    labels = np.unique(y_true)
    if (len(labels)==1):
        if labels[0] == 0:
            y_true = np.append(y_true, 1)
            y_proba = np.vstack( [y_proba, [[0,1]] ] )
        elif labels[0]==1:
            y_true = np.append(y_true, 0)
            y_proba = np.vstack( [y_proba, [[1, 0]] ] )
        else:
            raise ValueError("Only Support Binary Classification now!")
        
    not_vague_idx = np.sum(y_proba, axis=1)!=0
    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, -1]
    aupr = average_precision_score(y_true[not_vague_idx], y_proba[not_vague_idx])
    return aupr

metric_name_list = [
    "AUROC", 
    "AUPR",
    "F1",
    "ACC",
    'Precision',
    'Recall',
]

def scores_clf(y_true, proba):
    """
    return
    ------
        report: DataFrame, shape of (#)
    """
    pred = np.argmax(proba, axis=1)
    if proba.shape[1]>2:
        # 如果是多分类问题
        score_list = [ 
            roc_auc_score(y_true, proba, multi_class="ovr"),
            np.NaN,
            f1_score(y_true, pred, average="weighted"),
            accuracy_score(y_true ,pred),
            precision_score(y_true, pred, average="weighted"),
            recall_score(y_true, pred, average="weighted"),
        ]
    else:
        score_list = [ 
            roc_auc_score(y_true, proba[:, 1]),
            average_precision_score(y_true, proba[:, 1]),
            f1_score(y_true, pred),
            accuracy_score(y_true ,pred),
            precision_score(y_true, pred),
            recall_score(y_true, pred),
        ]
    score_name = [
        "AUROC", 
        "AUPR",
        "F1",
        "ACC",
        'Precision',
        'Recall',
    ]
    score_df = pd.DataFrame(score_list, index=score_name)
    return score_df

def collect_metrics(dir:str, n_view:int):
    """收集每种view的组合的结果，并存到新的csv文件中"""
    from itertools import combinations
    import os 
    result_df = pd.DataFrame()
    cols = ["MVUGCForest", "MVUGCForest_std"]
    for n_comb in range(2, n_view+1):
        for c in combinations(range(n_view), n_comb):
            # 视图特征组合
            str_name = str(c).replace(',', '_')
            result_path = dir+f"/{str_name}"
            filename = result_path + f"/MVUGCForest_{result_path.split('/')[-1]}_cv.csv"
            if not os.path.exists(result_path):
                raise FileNotFoundError(result_path, "Not Found")
            # 打印结果存储的文件夹
            # print(filename)
            df = pd.read_csv(filename, index_col=0)
            result_df[[col+f"_view_{str_name}" for col in cols]] = df[cols]
            if n_comb == n_view:
                # 保存单个view的结果
                single_view_cols = [col+f"_view_{i}" for i in range(n_view) for col in cols]
                result_df[single_view_cols] = df[single_view_cols]
    sorted_cols = sorted(result_df.columns, key=lambda x: (len(x), x))
    result_df = result_df[sorted_cols]
    result_df.to_csv(os.path.join(dir, f"{dir.split('/')[-1]}_cv.csv") )
    
if __name__ == "__main__":
    dir = "/data/tq/RECOMB2024/result/消融实验_apart/all_drop"
    collect_metrics(dir, 4)