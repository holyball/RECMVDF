
import numpy as np
from numpy import ndarray as arr
from sklearn import ensemble
from .layer import Layer
from .logger import get_logger, get_opinion_logger, get_custom_logger
from .node import NodeClassifier
from .base import BaseLearner
from .util_processpoolexecutor import *
from copy import deepcopy
from typing import List, Tuple, Any
# 为了保证可以完整打印opinion
np.set_printoptions(threshold=np.inf)

class View(BaseLearner):
    def __init__(self, layer_id, view_id, forest_params_list, **kwargs) -> None:
        """_summary_

        Args:
            layer_id (_type_): _description_
            view_id (_type_): _description_
            
            n_nodes (int): 森林节点数量
        """
        super().__init__(**kwargs)
        self.layer_id = layer_id
        self.view_id = view_id
        self.n_nodes = kwargs.get("n_nodes", 2)
        self._nodes = []

        self._init_nodes(forest_params_list)
        
    def _init_nodes(self, forest_params_dict=None):
        """初始化view中的nodes

        Args:
            forest_params_list (List[Dict]): 各个node的参数字典
        """
        if forest_params_dict is None:
            forest_params_dict = {}
            for _ in range(int(self.n_nodes/2)):
                forest_params_dict["RandomForestClassifier"] = {
                    "n_estimators": 10, 
                    # "max_features": None,   # None/'sqrt'/float
                    # "max_samples": 0.7,
                    "max_depth": None, 
                    "n_jobs": -1, 
                    # "min_samples_leaf": 2,
                    # "min_weight_fraction_leaf": 2/504,
                    "max_features": 'sqrt',
                    # "bootstrap": False,
                }
                forest_params_dict["ExtraTreesClassifier"] = {
                    "n_estimators": 10, 
                    # "max_features": None,   # None/'sqrt'/float
                    # "max_samples": 0.7,
                    "max_depth": None, 
                    "n_jobs": -1, 
                    # "min_samples_leaf": 2,
                    # "min_weight_fraction_leaf": 2/504,
                    "max_features": 'sqrt',
                    # "bootstrap": False,
                }
        for i, (forest_type, forest_param) in enumerate(forest_params_dict.items()):
            self._nodes.append(
                NodeClassifier(
                    self.layer_id, self.view_id, i, 
                    forest_type=forest_type, forest_params=forest_param) )
        
        
    def _fit_view(self, x, y, group_id=None):
        """私有函数, 训练当前的view
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(fit_node, node, x, y, group_id) for node in self._nodes]
            fitted_nodes = [future.result() for future in futures]
        self._nodes = fitted_nodes
        self._is_fitted = True
    
    def fit(self, x, y, group_id=None):
        self._fit_view(x, y, group_id=group_id)
        
    def evaluate(self, x, y, eval_func, return_proba=False):
        """评估训练集的结果, 有几种方法: 
            1. 直接调用node.evaluate()方法, 并将所有的评分取均值
            2. 收集node在验证集上的预测结果, 然后使用这个预测结果来评估

        Args:
            x (_type_): _description_
            y (_type_): _description_
            eval_func (_type_): _description_
        
        Returns:
            score:
            proba:
        """
        
        # 评估方法2:

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(evaluate_proba, node, x, y, eval_func) 
                for node in self._nodes
            ]
            y_proba_node_mat = [future.result() for future in futures]
        
        # ave
        y_proba_view = np.mean(y_proba_node_mat, axis=0)
        
        
        if return_proba:
            return y_proba_view
        return eval_func(y, y_proba_view)
    
    def generate_boosting_features(self, x, group_id=None):
        assert self._is_fitted, "This layer not fitted"

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(generate_boosting_features_func, node, x, group_id) 
                for node in self._nodes
            ]
            boosting_features = [future.result() for future in futures]
        boosting_features = np.hstack(boosting_features)
        return boosting_features
    
    def predict_proba(self, x, y=None):
        """预测x, 返回概率

        Args:
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
        """

        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(predict_node, node, x, y) for node in self._nodes]
            y_proba_node_mat = [future.result() for future in futures]
        
        # ave
        y_proba_view = np.mean(y_proba_node_mat, axis=0)
        return y_proba_view
        

# class View(object):
#     def __init__(self) -> None:
#         self.layers: List[Layer] = []
#         self.opinion_list: List[arr] = []
#         pass

#     def __len__(self):
#         """返回view中layer的数量"""
#         return len(self.layers)

#     def add_layer(self, layer: Layer):
#         self.layers.append(layer)
        
#     def predict_use_layer(self, x:arr, l:int, view_opinion_generation_method:str):
#         """使用指定的层作预测
#         先判断指定的层是否是最后一层, 如果是最后一层, 调用layer的_predict_opinion()方法
#         否则, 调用layer类的predict_opinion()方法
        
#         Parameters
#         ----------
#         x: ndarray
#             预测样本
#         l: int, l <= n_layer-1, 
#             指定的层数
#         view_opinion_generation_method: str, 'joint' or 'mean' or 'sum'

#         Return:
#         opinions:  (#nodes, #samples, #class+1) or (#samples, #class+1)
#         """
#         assert l<len(self), " l 必须小于层数"
#         if l == len(self)-1:
#             return self.layers[l]._predict_opinion_final_layer(x, view_opinion_generation_method)
#         else:
#             return self.layers[l].predict_opinion(x)

#     # def predict(self, x):
#     #     """使用所有层做预测, 输出每一层的预测产物
#     #     这里实现的困难点在于: 下一层的特征会依赖与上一层的结果, 
#     #     需要在每次预测之前调整特征
        
#     #     Returns:
#     #     -------
#     #     proba_list: ndarray of shape (#layers, #nodes, #samples, #classes)
#     #         第0维, 每一个元素都是对应层的预测概率
#     #     opinion_list: ndarray of shape (#layers, #nodes, #samples, #classes+1)
#     #     """
#     #     proba_list = []
#     #     opinion_list = []
#     #     for layer in self.layers:
#     #         probas = layer.predict_opinion(x)
        
#     def keep_layers(self, start: int, end: int):
#         """保留序号为从start到end的层, 左闭右开"""
#         self.layers = self.layers[start:end]


    
#     def build_boost_features(self, x, is_stacking: bool=False):
#         """为样本x创建增强特征(deprecated)
        
#         Parameters
#         ----------
#         x: ndarray, features
#         is_stacking: bool, 是否对增强特征进行堆叠, 默认False(不堆叠)

#         Returns
#         -------
#         boost_features: ndarray, 增强特征
#         """
#         if len(x) == 0:
#             return np.empty((0,1))
#         proba_list = []
#         opinion_list = []
#         x = np.copy(x)
#         n_features = x.shape[1]
#         for layer in self.layers:
#             proba, opinion, _ = layer.predict_opinion(x)
#             proba = np.hstack(proba)
#             opinion = np.hstack(opinion)
#             proba_list.append(proba)
#             opinion_list.append(opinion)
#             boost_features = proba

#             if is_stacking:
#                 x = np.hstack([x, boost_features])
#             else:
#                 x = np.hstack([x[:, :n_features], boost_features])
#         boost_features = x[:, n_features:]
        
#         return boost_features
