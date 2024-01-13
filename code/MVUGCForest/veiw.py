import numpy as np
from numpy import ndarray as arr
from sklearn import ensemble
from .layer import Layer
from .logger import get_logger, get_opinion_logger, get_custom_logger
from .node import NodeClassifier
# from .uncertainty import get_opinion, joint_multi_opinion, opinion_to_proba
from copy import deepcopy
from typing import List, Tuple, Any
# 为了保证可以完整打印opinion
np.set_printoptions(threshold=np.inf)

class View(object):
    def __init__(self) -> None:
        self.layers: List[Layer] = []
        self.opinion_list: List[arr] = []
        pass

    def __len__(self):
        """返回view中layer的数量"""
        return len(self.layers)

    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        
    def predict_use_layer(self, x:arr, l:int, view_opinion_generation_method:str):
        """使用指定的层作预测
        先判断指定的层是否是最后一层, 如果是最后一层, 调用layer的_predict_opinion()方法
        否则, 调用layer类的predict_opinion()方法
        
        Parameters
        ----------
        x: ndarray
            预测样本
        l: int, l <= n_layer-1, 
            指定的层数
        view_opinion_generation_method: str, 'joint' or 'mean' or 'sum'

        Return:
        opinions:  (#nodes, #samples, #class+1) or (#samples, #class+1)
        """
        assert l<len(self), " l 必须小于层数"
        if l == len(self)-1:
            return self.layers[l]._predict_opinion_final_layer(x, view_opinion_generation_method)
        else:
            return self.layers[l].predict_opinion(x)

    # def predict(self, x):
    #     """使用所有层做预测, 输出每一层的预测产物
    #     这里实现的困难点在于: 下一层的特征会依赖与上一层的结果, 
    #     需要在每次预测之前调整特征
        
    #     Returns:
    #     -------
    #     proba_list: ndarray of shape (#layers, #nodes, #samples, #classes)
    #         第0维, 每一个元素都是对应层的预测概率
    #     opinion_list: ndarray of shape (#layers, #nodes, #samples, #classes+1)
    #     """
    #     proba_list = []
    #     opinion_list = []
    #     for layer in self.layers:
    #         probas = layer.predict_opinion(x)
        
    def keep_layers(self, start: int, end: int):
        """保留序号为从start到end的层, 左闭右开"""
        self.layers = self.layers[start:end]


    
    def build_boost_features(self, x, is_stacking: bool=False):
        """为样本x创建增强特征(deprecated)
        
        Parameters
        ----------
        x: ndarray, features
        is_stacking: bool, 是否对增强特征进行堆叠, 默认False(不堆叠)

        Returns
        -------
        boost_features: ndarray, 增强特征
        """
        if len(x) == 0:
            return np.empty((0,1))
        proba_list = []
        opinion_list = []
        x = np.copy(x)
        n_features = x.shape[1]
        for layer in self.layers:
            proba, opinion, _ = layer.predict_opinion(x)
            proba = np.hstack(proba)
            opinion = np.hstack(opinion)
            proba_list.append(proba)
            opinion_list.append(opinion)
            boost_features = proba

            if is_stacking:
                x = np.hstack([x, boost_features])
            else:
                x = np.hstack([x[:, :n_features], boost_features])
        boost_features = x[:, n_features:]
        
        return boost_features
    

    
    