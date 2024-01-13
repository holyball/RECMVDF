from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaseEnsemble
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import euclidean_distances
from typing import Tuple, Dict
from numpy import ndarray as arr
from copy import deepcopy
from .util import accuracy, f1_macro, auroc, aupr
from .logger import get_logger, get_custom_logger
from .uncertainty import *
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
import os
from .base import BaseLearner
'''
此处导入新的包
'''

class NodeClassifier(BaseLearner):
    def __init__(self, layer_id, view_id, index, forest_type, forest_params,
                 **kwargs):
                #  config, random_state, logger_path=None):
        """
            layer_id: int
            view_id: int
            index: int, ID of unit
            forest_type: str, 森林类型
            forest_params: dict, 森林的参数
        """
        self.fitted = False      
        self.forest_type: str = forest_type
        self.forest_params = forest_params
        self.name = "layer_{}, view_{}, estimstor_{}, {}".format(
            layer_id, view_id, index, forest_type)
        # print("node random_state: ", self.random_state)
        self.n_fold = kwargs.get("n_fold", 5)
        self.estimator_class = globals()[forest_type]
        self.des = kwargs.get("is_des", False)  # 是否动态集成选择
        self.n_trees = kwargs.get("n_trees", 50)

        self.estimators_: Dict[int, BaseEnsemble] = {i:None for i in range(self.n_fold)}
        self.n_class = None
        self.neighs_ = [None for _ in range(self.n_fold)]
        self.train_labels = [None for _ in range(self.n_fold)]
        self.n_feature_origin: int                                  # 原始特征的数量
        self.n_sample_origin: int

        # 日志记录
        self.logger_path = kwargs.get("logger_path", "./MVUGCForest_info")
        assert os.path.exists(self.logger_path), f"logger_path: {self.logger_path} not exist! "
        self.logger = get_custom_logger("KFoldWrapper", "Node_train_log.txt", self.logger_path)

    def _init_estimator(self):
        """初始化基本森林学习器"""
        if self.forest_type in ["XGBClassifier", "LGBMClassifier"]:
            assert self.forest_params.get("num_class"), "请为XGBoost分类器设置 'num_class' 参数"
            
        est_args = self.forest_params.copy()
        if self.random_state:
            est_args["random_state"]=self.random_state
        return self.estimator_class(**est_args)

    def fit(self, 
            x, y, sample_weight:arr=None, 
            n_sample_origin: int = None,
            ) -> Tuple[arr, arr]:
        """
        
        Returns
        -------
        y_proba_wrapper: ndarray, shape is (n_samples, n_classes)
            验证集概率
        opinion_wrapper: ndarray, shape is (n_samples, n_classes+1)
            结点的opinion
        evidence_wrapper: ndarray or None, ndarray shape is (#samples, n_classes). 
            仅当使用基于证据计算不确定性时, 该返回值才是ndarray, 否则返回None
        """
        self.logger.info(f"----------------------------------------------------------------------------------------------")
        skf = StratifiedKFold(n_splits=self.n_fold,
                              shuffle=True, random_state=self.random_state)
        cv = [(t, v) for (t, v) in skf.split(x, y)]

        n_sample = len(y)
        n_class = len(np.unique(y))
        self.n_class = n_class

        self._fit_estimators(x=x, y=y, cv=cv, sample_weight=sample_weight)

        y_proba = np.empty(shape=(n_sample, n_class))
        for k in range(self.n_fold):
            est = self.estimators_[k]
            train_id, val_id = cv[k]

            if self.forest_type in ["RandomForestClassifier", "ExtraTreesClassifier"]:
                # 如果是并行树
                val_proba = predict_proba_mat_rf(est, x[val_id])

            y_proba[val_id] = val_proba

            self.logger.info(
                "{}, n_fold_{}, Accuracy={:.4f}, f1_score={:.4f}, auroc={:.4f}, aupr={:.4f}".format(
                self.name, k, accuracy(y[val_id], val_proba), f1_macro(y[val_id], val_proba), 
                auroc(y[val_id], val_proba), aupr(y[val_id], val_proba)))

        self.logger.info(
            "{}, {},Accuracy={:.4f}, f1_score={:.4f}, auroc={:.4f}, aupr={:.4f}".format(
                self.name, "wrapper", accuracy(y, y_proba), f1_macro(y,y_proba),
                auroc(y, y_proba), aupr(y, y_proba) ) )
        self.logger.info("----------")

        self.cv = [(t[t<n_sample_origin], v[v<n_sample_origin]) for t, v in cv]
        self.X_train = x[:n_sample_origin]
        self.y_train = y[:n_sample_origin]
        self.fitted = True  # 是否拟合的标志
        return y_proba

    def predict_proba(self, x_test) -> arr:
        proba = 0
        for est in self.estimators_:
            proba += est.predict_proba(x_test)
        proba /= len(self.estimators_)
        return proba
    
    def _fit_estimators(self, x, y, cv, sample_weight=None):
        """拟合基学习器(森林)"""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        for k in range(self.n_fold):
            est = self._init_estimator()
            train_id, val_id = cv[k]
            # print(x[train_id])
            est.fit(x[train_id], y[train_id], sample_weight=sample_weight[train_id])
            self.estimators_[k] = est

    def _fit_neighbors(self, x, y, cv, r=None):
        """基于预测结果训练为每一个森林训练Nearest Neighbors"""
        
        if r is None:
            if hasattr(self.estimators_[0], "n_estimators"):
                n_tree = self.estimators_[0].n_estimators
            else:
                n_tree = len(self.estimators_[0].estimators_)
            r = np.sqrt(self.n_trees / 5)
        for i, (train_idx, val_idx) in enumerate(cv):
            prediction_mat = self._apply_tree_prediction(x[val_idx], i)

            # 如果是多分类, 对prediction_mat做onehot编码
            if self.n_class > 2:
                ohe = OneHotEncoder(sparse=False)
                prediction_mat = ohe.fit_transform(prediction_mat)
            neigh = NearestNeighbors(radius=r).fit(prediction_mat)
            self.neighs_[i] = neigh
            self.train_labels[i] = y[val_idx]

    def _apply_tree_prediction(self, x_for_predict, forest_id):
        """针对待预测样本, 获取森林(序号为forest_id)中所有树的预测结果
        
        returns
        -------
        prediction_mat: ndarray of shape (#samples, #trees)"""
        prediction_mat = get_trees_predict(self.estimators_[forest_id], x_for_predict)
        return prediction_mat
    
    def _apply_tree_predict_proba(self, x_for_predict, forest_id):
        """针对待预测样本, 获取森林(序号为forest_id)中所有树的预测结果
        
        returns
        -------
        proba_mat: ndarray of shape (#samples, #trees * #classes)
        """
        proba_mat = get_trees_predict_proba(self.estimators_[forest_id], x_for_predict)
        return proba_mat
    
    def _normalize_log_density(self, x, y):
        return ReLU(x-y)/np.abs(x)
        # return ReLU(x-y)/100



def get_trees_predict(forest, X):
    """获取森林中每一棵树的预测结果
    Parameters
    ----------
    forest: estimator
    X: array_like
    
    Returns
    -------
    prediction_mat: ndarray of shape (#samples, #trees)
    """
    if hasattr(forest, "estimators_"):
        prediction_mat = np.transpose([tree.predict(X) for tree in forest.estimators_])
    else:
        print("请传入支持estimators_属性的森林")
    return prediction_mat

def get_trees_predict_proba(forest, X):
    """获取森林中每一棵树的预测结果的概率
    Parameters
    ----------
    forest: estimator
    X: array_like
    
    Returns
    -------
    proba_mat: ndarray of shape (#samples, #trees * #classes)
    """
    if hasattr(forest, "estimators_"):
        proba_mat = np.hstack([tree.predict_proba(X) for tree in forest.estimators_])
    else:
        print("请传入支持estimators_属性的森林")
    return proba_mat
    
def predict_proba_mat_rf(ensemble, X) -> arr:
    """计算并行森林的概率矩阵
    Returns
    -------
    proba_mat: ndarray of shape(#forests, #samples, #classes)"""
    proba_mat = []
    for tree in ensemble.estimators_:
        proba_mat.append(tree.predict_proba(X))
    return np.array(proba_mat)

def get_precision_f1(y_true, y_pred):
    """ 获取 每个类的精度 和 整体的f1
    
    Returns
    -------
    record: DataFrame, shape = (#classes + 1, ).
        index = ['precision_class_0', 'precision_class_1', ..., 'f1-score']
    """
    from sklearn.metrics import classification_report
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    precision_list = []
    labels = np.unique(y_true)
    for label in labels:
        precision_list.append(report_dict[str(label)]['precision'])
    precision_list.append(report_dict['macro avg']['f1-score'])
    index = [f"precision_class_{label}" for label in labels]
    index.append('f1-score')
    return pd.DataFrame(precision_list, index=index)


