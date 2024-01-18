import numpy as np
import pandas as pd
import random
from copy import deepcopy
from typing import List, Tuple, Dict
from numpy import ndarray as arr 
from sklearn.metrics import euclidean_distances
import os

def _construct_resampling_set(x, y, label, tags, fold, random_state=None):
    rng = np.random.RandomState(seed=random_state)

    hard_c_mask = (y==label) & (tags==3)
    hard_not_c_idx = np.unique(    # 类标签不为c类的困难样本索引
        rng.choice( np.argwhere(y!=label).squeeze(), int(np.sum(hard_c_mask)*fold) ) )
    if np.sum(hard_c_mask) < 6: return np.empty((0, x.shape[1])), np.empty(0)    # 如果类别对应的困难样本量小于6, 则不采样
    x_to_sample = np.vstack( (x[hard_c_mask], x[hard_not_c_idx]) )
    y_to_sample = np.hstack( (y[hard_c_mask], y[hard_not_c_idx]) )
    
    return x_to_sample, y_to_sample
        
def _resample_hard(x, y, tags, fold, kind='random', random_state=None, **kwargs):
    """困难样本重采样
    
    Parameters
    ----------
    x (ndarray): shape of (#samples, #features)
    y (ndarray): shape of (#samples, )
    fold (float): 采样倍率
    random_state (int):
    """
    rng = np.random.RandomState(seed=random_state)
    labels, sample_num = np.unique(y, return_counts=True)
    n_class = len(labels)
    x_new_list, y_new_list = [], []
    
    for c in labels:
        # 为每个类别的困难样本设置采样数据集
        x_to_sample, y_to_sample = _construct_resampling_set(
            x, y, c, tags, fold, random_state )
        # 开始采样
        if np.all(np.unique(y_to_sample, return_counts=True)[1] < 6):
            return np.empty((0, x_to_sample.shape[1])), np.empty(0)
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
            x_res, y_res = sm.fit_resample(x_to_sample, y_to_sample)
            x_new_, y_new_ = x_res[len(x_to_sample):], y_res[len(x_to_sample):]
            
        except ValueError:
            print(f"困难样本采样失败啦, 困难样本的分布为: {np.unique(y_to_sample, return_counts=True)}")
            # print(np.unique(y, return_counts=True)[1])
            x_new_ = np.empty((0, x.shape[1]))
            y_new_ = np.empty(0)
        
        x_new_list.append(x_new_)
        y_new_list.append(y_new_)
    x_new = np.vstack(x_new_list)
    y_new = np.hstack(y_new_list)
    
    return x_new, y_new
    

def _resample_outlier(x, y, tags, k_neighbours, random_state=None):
    """为离群样本采样

    Args:
        x (ndarray): _description_
        y (ndarray): _description_
        tags (ndarray): _description_
        k_neighbours (int): 希望的同类最近邻样本数量
    """
    rng = np.random.RandomState(random_state)
    x_new = []
    y_new = []
    
    ids = np.argwhere(tags).reshape(-1)
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

def resample(x, y, tags, random_state=None, **kwargs):
    
    fold = kwargs.get("hard_fold", None)
    kind = kwargs.get("hard_resample_kind", "random")
    k_neighbours = kwargs.get("outlier_k_neighbours")
    x_new_hard, y_new_hard = _resample_hard(x, y, tags, fold, kind = kind, 
                                            random_state=random_state)
    
    x_new_outlier, y_new_outlier = _resample_outlier(x, y, tags, k_neighbours=k_neighbours,
                                                     random_state=random_state)

    x_new = np.vstack([x_new_hard, x_new_outlier])
    y_new = np.hstack([y_new_hard, y_new_outlier])
    return x_new, y_new

def _get_confidence(proba):
    return np.argmax(proba, axis=1).squeeze()

def get_confidence(proba_list):
    """_summary_

    Args:
        proba_list (_type_): _description_

    Returns:
        ndarray: shape (n_samples, n_views)
    """
    confidece_list = np.transpose([_get_confidence(proba) for proba in proba_list])
    return confidece_list

def get_prediction(proba_list):
    y_pred = np.transpose([np.argmax(proba, axis=1) for proba in proba_list])
    return y_pred

def _assign_tags_training(confidence_list, y_pred, y_true) -> arr:
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
    
    mean_confidence = np.mean(confidence_list, axis=0)
    greater_mask = confidence_list>mean_confidence
    high_confidence_mask = np.all(greater_mask, axis=1)
    
    correct_mask = np.all(y_pred==y_true.reshape((-1, 1)), axis=1)
    
    marks = np.ones_like(y_true)
    marks[high_confidence_mask & correct_mask] = 2
    marks[(~high_confidence_mask) & (~correct_mask)] = 3
    marks[high_confidence_mask & (~correct_mask)] = 4
    
    return marks, mean_confidence
  
def _assign_tags_prediction(confidence_list: List[arr], confidence_thresholds: List):
    """在测试阶段划分样本类型
    """
    greater_mask = confidence_list>confidence_thresholds
    high_confidence_mask = np.all(greater_mask, axis=1)
    
    tags = np.ones(len(confidence_list))
    tags[high_confidence_mask] = 2
    
    return tags

def assign_tags(proba_list, **kwargs):
    """assign tags base span-strategy

    Args:
        proba_list (ndarray): _description_
        y_true (ndarray): optional, refer to training stage
        confidence_thresholds (ndarray): optional, refer to predicting stage
    Returns:
        ndarray: tags
    """
    y_true = kwargs.get("y_true", None)
    confidence_thresholds = kwargs.get("confidence_thresholds", None)
    
    confidence_list = get_confidence(proba_list)
    
    if y_true is not None:
        y_pred = get_prediction(proba_list)
        tags, threshold_list = _assign_tags_training(confidence_list, y_pred, y_true)
        return tags, threshold_list
    if confidence_thresholds is not None:
        return _assign_tags_prediction(confidence_list, confidence_thresholds)
      