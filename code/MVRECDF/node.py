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
        super().__init__(**kwargs)
          
        self.forest_type: str = forest_type
        self.forest_params = forest_params
        self.name = "layer_{}, view_{}, estimstor_{}, {}".format(
            layer_id, view_id, index, forest_type)
        # print("node random_state: ", self.random_state)
        self.n_fold = kwargs.get("n_fold", 5)
        self.forest = globals()[forest_type]
        self.des = kwargs.get("is_des", False)  # 是否动态集成选择
        self.n_trees = kwargs.get("n_trees", 50)

        self.estimators_: List = [None for _ in range(self.n_fold)]
        self.n_class = None
        self.neighs_ = [None for _ in range(self.n_fold)]
        self.train_labels = [None for _ in range(self.n_fold)]
        self.n_feature_origin: int                                  # 原始特征的数量
        self.n_sample_origin: int
        
        self.cv_train = None

        # 日志记录
        self.logger_path = kwargs.get("logger_path", "./MVRECDF_info")
        assert os.path.exists(self.logger_path), f"logger_path: {self.logger_path} not exist! "
        self.logger = get_custom_logger("KFoldWrapper", "Node_train_log.txt", self.logger_path)

    def _init_estimator(self):
        """初始化基本森林学习器"""
        if self.forest_type in ["XGBClassifier", "LGBMClassifier"]:
            assert self.forest_params.get("num_class"), "请为XGBoost分类器设置 'num_class' 参数"
            
        est_args = self.forest_params.copy()
        if self.random_state:
            est_args["random_state"]=self.random_state
        return self.forest(**est_args)
    
    def _get_cv_train(self, x, y, group_id):
        if group_id is None:
            skf = StratifiedKFold(n_splits=self.n_fold,
                shuffle=True, random_state=self.random_state)
            cv_train = [(t, v) for (t, v) in skf.split(x, y)]
        else:
            cv_train = []
            # print(f"group_id len: {len(group_id)}")
            for k in range(self.n_fold):
                train_id = np.argwhere((group_id!=k) | (group_id==-1)).squeeze()
                val_id = np.argwhere(group_id==k).squeeze()
                cv_train.append((train_id, val_id))
                # print(f"cv_train_{k}: {len(train_id)+len(val_id)}")
        self.cv_train = cv_train

    def fit(self, 
            x, y, sample_weight:arr=None, 
            n_sample_origin: int=None,
            group_id: List=None,
            ) -> Tuple[arr, arr]:

        self.logger.info(f"----------------------------------------------------------------------------------------------")
        n_sample = len(x)
        n_class = len(np.unique(y))
        self.n_class = n_class
        
        self._get_cv_train(x, y, group_id)
    
        self._fit_estimators(x=x, y=y, sample_weight=sample_weight)

        # try:
        #     self._fit_estimators(x=x, y=y, sample_weight=sample_weight)
        # except Exception as e:
        #     print(e)
        #     # x_hasNaN = np.isnan(x).any()
        #     # print('X包含 NaN: ', x_hasNaN)
        #     # y_hasNaN = np.isnan(y).any()
        #     # print("y包含 NaN: ", y_hasNaN)
        #     print(f"x.shape: {x.shape}, y.shape:{y.shape}")
            
        #     pd.DataFrame(x).to_csv(f"x_new_{self.name}.csv")
        #     pd.DataFrame(y).to_csv(f"y_new_{self.name}.csv")
        #     exit()
        # y_proba = np.empty(shape=(n_sample, n_class))
        # for k in range(self.n_fold):
        #     est = self.estimators_[k]
        #     _, val_id = self.cv_train[k]
        #     if self.forest_type in ["RandomForestClassifier", "ExtraTreesClassifier"]:
        #         # 如果是并行树
        #         val_proba = predict_proba_mat_rf(est, x[val_id])

        #     y_proba[val_id] = val_proba

        #     self.logger.info(
        #         "{}, n_fold_{}, Accuracy={:.4f}, f1_score={:.4f}, auroc={:.4f}, aupr={:.4f}".format(
        #         self.name, k, accuracy(y[val_id], val_proba), f1_macro(y[val_id], val_proba), 
        #         auroc(y[val_id], val_proba), aupr(y[val_id], val_proba)))

        # self.logger.info(
        #     "{}, {},Accuracy={:.4f}, f1_score={:.4f}, auroc={:.4f}, aupr={:.4f}".format(
        #         self.name, "wrapper", accuracy(y, y_proba), f1_macro(y,y_proba),
        #         auroc(y, y_proba), aupr(y, y_proba) ) )
        # self.logger.info("----------")

        self._is_fitted = True  # 是否拟合的标志
        # return y_proba

    def evaluate(self, x, y, eval_func, return_proba=False):
        assert self._is_fitted, "This node not fitted"
        y_proba = np.empty((x.shape[0], self.n_class))
        
        for i, (_, val_idx) in enumerate(self.cv_train):
            y_proba[val_idx] = self.estimators_[i].predict_proba(x[val_idx])
        
        if return_proba:
            return y_proba
        return eval_func(y, y_proba)
    
    def generate_boosting_features(self, x, group_id):
        x_train = x[ np.argwhere(group_id!=-1).squeeze()]
        x_predict = x[ np.argwhere( (group_id==-1) ).squeeze() ] 
        
        if len(x_train) > 0:
            boost_feature_train = self.evaluate(x_train, None, None, return_proba=True)
        if len(x_predict) > 0:
            boost_feature_list = []
            for forest in self.estimators_:
                boost_feature_list.append(forest.predict_proba(x_predict))
                boost_feature_predict = np.mean(boost_feature_list, axis=0)
        
        if len(x_train) == 0:
            return boost_feature_predict
        elif len(x_predict) == 0:
            return boost_feature_train
        else:
            return np.vstack([boost_feature_train, boost_feature_predict])
    
    # def generate_boosting_features(self, x, group_id=None):
    #     assert self._is_fitted, "This node not fitted"
    #     if group_id is None:
    #         # 训练阶段的增强特征
    #         return self.evaluate(x, None, None, return_proba=True)
    #     else:
    #         # 预测阶段的增强特征
    #         boost_features = []
    #         for forest in self.estimators_:
    #             boost_features.append(forest.predict_proba(x))
    #         return np.mean(boost_features, axis=0)
            
    def predict_proba(self, x, y=None) -> arr:
        proba = 0
        for est in self.estimators_:
            proba += est.predict_proba(x)
        proba /= len(self.estimators_)
        return proba
    
    def _fit_estimators(self, x, y, sample_weight=None):
        """拟合基学习器(森林)"""
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        for k in range(self.n_fold):
            est = self._init_estimator()
            train_id, _ = self.cv_train[k]
            est.fit(x[train_id], y[train_id], sample_weight=sample_weight[train_id])
            # try:
            #     est.fit(x[train_id], y[train_id], sample_weight=sample_weight[train_id])
            # except ValueError:
            #     print(np.shape(x[train_id]), np.shape(y[train_id]))
            #     pd.DataFrame(x[train_id]).to_csv()
            # except TypeError:
            #     # print(train_id)
            #     # print(np.shape(x[train_id]), np.shape(y[train_id]))
            #     print(np.shape(y))
            #     exit()
            self.estimators_[k] = est

    def _fit_neighbors(self, x, y, r=None):
        """基于预测结果训练为每一个森林训练Nearest Neighbors"""
        
        if r is None:
            if hasattr(self.estimators_[0], "n_estimators"):
                n_tree = self.estimators_[0].n_estimators
            else:
                n_tree = len(self.estimators_[0].estimators_)
            r = np.sqrt(self.n_trees / 5)
        for i, (val_idx) in enumerate(self.cv_train):
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


