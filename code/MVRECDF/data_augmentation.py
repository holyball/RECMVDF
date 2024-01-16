import numpy as np
import pandas as pd
import random
from copy import deepcopy
from typing import List, Tuple, Dict
from numpy import ndarray as arr 
from sklearn.metrics import euclidean_distances
import os

def resample_hard(x, y, kind='random', random_state=None):
    """重采样
    
    Parameters
    ----------
    x (ndarray): shape of (#samples, #features)
    y (ndarray): shape of (#samples, )
    random_state (int):
    """
    if len(np.unique(y)) == 1:
        # print(y)
        return np.empty((0, x.shape[1])), np.empty(0)
    if np.all(np.unique(y, return_counts=True)[1] < 6):
        return np.empty((0, x.shape[1])), np.empty(0)
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE
        if kind == 'random':
            sm = RandomOverSampler()
        elif kind == 'smote':
            sm = SMOTE(sampling_strategy='not majority')
        elif kind == 'borderlinesmote':
            sm = BorderlineSMOTE(kind='borderline-2')
        if random_state:
            sm.set_params(random_state=random_state)
        x_res, y_res = sm.fit_resample(x, y)
        x_new, y_new = x_res[len(x):], y_res[len(x):]

        return x_new, y_new
    except ValueError:
        print(f"困难样本采样失败啦, 困难样本的分布为: {np.unique(y, return_counts=True)}")
        # print(np.unique(y, return_counts=True)[1])
        return np.empty((0, x.shape[1])), np.empty(0)

def resample_outlier(x, y, mask, k_neighbours, **kwargs):
    """为离群样本采样

    Args:
        x (ndarray): _description_
        y (ndarray): _description_
        mask (ndarray): _description_
        k_neighbours (int): 希望的同类最近邻样本数量
    """
    random_state = kwargs.get("random_state", None)
    rng = np.random.RandomState(random_state)
    x_new = []
    y_new = []
    
    ids = np.argwhere(mask).reshape(-1)
    for i in ids:
        instance = x[i]
        instance_label = y[i]
        
        # 计算 instance 的 k 近邻样本索引
        distances = np.linalg.norm(x - instance, axis=1)
        nearest_indices = np.argsort(distances)[1:k_neighbours+1]  # 不包括自身
        
        # 统计 k 近邻样本中的异类样本数量
        n_opposite = np.sum(y[nearest_indices] != instance_label)
        
        if n_opposite > 0:
            # 计算采样步长的上限
            step_upper = distances[nearest_indices[n_opposite-1]]
            
            if step_upper < 1e-3 or step_upper == np.NaN:
                # 如果步长上限过小或无效，不进行采样
                continue
            
            # 生成新样本
            steps = rng.uniform(1e-3, step_upper, n_opposite).reshape((-1, 1))
            diffs = x[nearest_indices[:n_opposite]] - instance
            x_res = np.atleast_2d(instance) + steps * diffs
            y_res = np.full(n_opposite, instance_label)
            
            x_new.append(x_res)
            y_new.append(y_res)
    
    if len(x_new) > 0:
        x_new = np.vstack(x_new)
        y_new = np.hstack(y_new)
    
    return x_new, y_new

def _get_confidence(proba):
    return np.argmax(proba, axis=1).squeeze()

def get_confidence(proba_list):
    confidece_list = np.transpose([_get_confidence(proba) for proba in proba_list])
    return confidece_list

def get_prediction(proba_list):
    y_pred = np.transpose([np.argmax(proba, axis=1) for proba in proba_list])
    return y_pred

def assign_tags(confidence_list, y_pred, y_true) -> arr:
    """划分样本
    对训练集: 同时考虑uncertainty 和 预测结果

    Parameters
    ----------
    confidence_list (ndarray):
    y_pred (ndarray): 
    y_true (ndarray): shape (#samples, ), 样本的真实标签
    
    Returns
    -------
    marks: ndarray of shape (#samples, ). 样本的标记, 包含了1, 2, 3, 4.
        每一个样本可能的标记有 "normal"-1, "easy"-2, "hard"-3, "outlier"-4
    -------
    """
    # y_true = np.array([1, 1, 0])
    # proba_list = np.array([[[1.0, 0.0], [0.3, 0.7], [0.6, 0.4]],
    #                        [[0.3, 0.8], [0.3, 0.7], [0.6, 0.4]]])
    
    # cofidence_list = get_confidence(proba_list)
    
    mean_confidence = np.mean(confidence_list, axis=0)
    greater_mask = confidence_list>mean_confidence
    high_confidence_mask = np.all(greater_mask, axis=1)
    
    correct_mask = np.all(y_pred==y_true.reshape((-1, 1)), axis=1)
    
    marks = np.ones_like(y_true)
    marks[high_confidence_mask & correct_mask] = 2
    marks[(~high_confidence_mask) & (~correct_mask)] = 3
    marks[high_confidence_mask & (~correct_mask)] = 4
    
    return marks, mean_confidence

def group_samples_for_test(confidence_list: List[arr], confidence_thresholds: List):
    """在测试阶段划分样本类型
    """
    greater_mask = confidence_list>confidence_thresholds
    high_confidence_mask = np.all(greater_mask, axis=1)
    
    marks = np.ones(len(confidence_list))
    marks[high_confidence_mask] = 2
    
    return marks