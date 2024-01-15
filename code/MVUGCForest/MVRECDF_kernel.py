'''
Description:     
Author: tanqiong
Date: 2023-04-18 17:09:34
LastEditTime: 2023-10-09 18:17:06
LastEditors: tanqiong
'''
import numpy as np
import pandas as pd
from .logger import get_logger, get_opinion_logger, get_custom_logger, mkdir
from .veiw import View
from .multiview_layer import Layer
from .util import df_to_csv, get_stage_matrix, get_scores, metrics_name
from .dataloader import DataLoader
from .evaluation import accuracy, f1_macro, mse_loss, aupr, auroc
from .base import BaseLearner

from copy import deepcopy
from typing import List, Tuple, Any, Dict, Union
import os
import warnings
warnings.filterwarnings("ignore")
# 为了保证可以完整打印opinion
np.set_printoptions(threshold=np.inf)


class MVRECForestClassifier(BaseLearner):

    def __init__(self, n_view, n_tree, n_fold, **kwargs):
        super().__init__(**kwargs)
        self.n_view = n_view
        self.n_fold = n_fold

        # boost features, 
        self.is_stacking_for_boost_features = kwargs.get("is_stacking_for_boost_features", False)  # 是否累加概率增强特征

        # cluster samples
        self.cluster_samples_layer = kwargs.get("cluster_samples_layer", True) 
        self.ta:float = kwargs.get("ta", 0.7)           # DBC的阈值参数
        self.alpha: float = kwargs.get("alpha", 0.7)    # gcForest_cs的阈值参数

        # resample
        self.is_resample = kwargs.get("is_resample", True)
        self.onn = kwargs.get("onn", 3)     # outlier nearest neighbour
        self.layer_resample_method = kwargs.get("layer_resample_method", "integration")   # "integration", "respective"
        self.accumulation = kwargs.get("accumulation", False)  # 人工样本是否累加

        # training
        self.span = kwargs.get("span", 2)
        self.max_layers = kwargs.get("max_layers", 10)
        self.early_stop_rounds = kwargs.get("early_stop_rounds", 2)
        self.is_weight_bootstrap = kwargs.get("is_weight_bootstrap", True) # 加权 bootstrap
        self.is_save_model = kwargs.get("is_save_model", False)
        self.eval_func = kwargs.get("eval_func", accuracy)
        
        # prediction
        self.bypass_prediction = kwargs.get("bypass_prediction", True)
        self.bypass_threshold = kwargs.get("bypass_threshold", 0.6)
        self.is_des = kwargs.get("is_des", False)

        # some variables for training or prediction
        self.n_view: int
        self.n_class: int
        self.bypass_weights = []
        self._layers = []

        # 记录中间结果的变量
        self._n_origin_features = []
        self.best_layer_id: int # 最佳层的层号

        # init logger
        self.logger_path = kwargs.get("logger_path", ".")
        self._check_logger_path()
        self._register_logger()
        
        # # 记录参数配置
        # self.train_logger.info(str(config))
        # self.predict_logger.info(str(config))
    
    def _check_logger_path(self):
        logger_path = self.logger_path
        assert os.path.exists(logger_path), "日志文件夹不存在! "
        if logger_path.find("MVRECDF_info") == -1:
            logger_path = os.path.join(logger_path, "MVRECDF_info")
        if not os.path.exists(logger_path):
            try: 
                os.mkdir(logger_path)
            except Exception: 
                raise Exception("路径错误")
        self.logger_path = logger_path
    
    def _register_logger(self):
        """注册LOGGER"""
        logger_path = self.logger_path
        self.train_logger = get_custom_logger(
            "RECMVDF_fit", "RECMVDF_fit.txt", logger_path)
        self.predict_logger = get_custom_logger(
            "RECMVDF_predict", "RECMVDF_predict.txt", logger_path)
    
    def fit(self, x_list, y, **kwargs):
        self._fit_MVRECDF(x_list, y, **kwargs)

    def _fit_MVRECDF(self, x_list, y, **kwargs):
        
        n_origin_sample = y.shape[0]
        self._n_origin_features = [x.shape[1] for x in x_list]
        y_origin = y.copy()
        sample_weight = np.ones(n_origin_sample)
        weight_bais = None    # 权重偏移(置信度低的样本, 权重大)
        
        layer_id = 0
        best_layer_id = 0
        best_score = 0
        while layer_id < self.max_layers:
            bypass_weights = []    # bypass加权权重
            
            # Weighting Bootstrap
            if self.is_weight_bootstrap and weight_bais is not None:
                sample_weight[:n_origin_sample] += weight_bais
            
            # Adding Boosting Features
            x_list = self._generate_next_layer_data(layer_id, x_list, None)
            
            layer = Layer(layer_id, self.n_view)
            group_id = self._split_train_data(x_list, y, self.random_state)
            layer.fit(x_list, y, group_id=group_id)
            

            eval_score = layer.evaluate(x_list, y, eval_func=self.eval_func)
            if (eval_score > best_score):
                best_layer_id = layer_id
                best_score = eval_score
                
            print(f"**** Train layer {layer_id} ****")
            print(f"eval score: {eval_score}, {self.eval_func.__name__}")
            print(f"best layer: {best_layer_id}, best score: {best_score}")
            
            if layer_id-best_layer_id >= self.early_stop_rounds:
                self._layers = self._layers[:best_layer_id+1]
                break
            self._layers.append(layer)

            layer_id += 1
            
    def _generate_next_layer_data(self, layer_id, x_list, group_id=None):
        if (layer_id==0):
            return x_list
        boosting_features = self._layers[layer_id-1].generate_boosting_features(x_list, group_id)
        if self.is_stacking_for_boost_features:
            x_list_new = [
                np.hstack([x, boost]) for x, boost in zip(x_list, boosting_features)
            ]
        else:
            x_list_new = [
                np.hstack(
                    [ x_list[i][:, :self._n_origin_features[i]], boosting_features[i] ] ) 
                for i in range(self.n_view)
            ]
        return x_list_new
    
    def _split_train_data(self, x_list, y, random_state=None):
        """将训练集划分为n_fold份, 用一个ndarray标记出每个样本是哪个fold的验证集

        Args:
            x_list (List): _description_
            y (ndarray): _description_
            random_state (int or None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(
            n_splits=self.n_fold, shuffle=True, random_state=random_state)
        gen = skf.split(x_list[np.random.randint(0, self.n_view)], y)
        val_idx_fold = np.empty(y.shape[0])
        for fold_idx, (_, val_idx) in enumerate(gen):
            val_idx_fold[val_idx] = fold_idx
        return val_idx_fold

    def predict_proba(self, x_list, y=None):
        y_proba_mat = []
        eval_score = 0
        best_layer_id = 0
        best_score = 0
        for layer_id in range(len(self._layers)):
            x_list = self._generate_next_layer_data(layer_id, x_list, "predict")
            y_proba_layer = self._layers[layer_id].predict_proba(x_list)
            y_proba_mat.append(y_proba_layer)
            
            if y is not None:
                eval_score = self.eval_func(y, y_proba_layer)
                if (eval_score > best_score):
                    best_layer_id = layer_id
                    best_score = eval_score
                print(f"**** Predict: layer {layer_id} ****")
                print(f"eval score: {eval_score}, {self.eval_func.__name__}")
                print(f"best layer: {best_layer_id}, best score: {best_score}")
            
        # if self.weighting_predict
        return np.mean(y_proba_mat, axis=0)