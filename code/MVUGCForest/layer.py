import numpy as np 
from .node import NodeClassifier
from .uncertainty import joint_multi_opinion
from typing import List
from numpy import ndarray as arr

class Layer(object):
    def __init__(self,layer_id, view_id=0, logger_path=None):
        self.layer_id: int = layer_id
        self.nodes:List[NodeClassifier] = []
        self.view_id: int = view_id
        self.logger_path = logger_path
        self.feature_names_ = []
        self.feature_importances_ = []

    def set_feature_names_(self, boost_feature_names):
        self.feature_names_ = [f"origin_{i}" for i in range(len(self.feature_importances_))]
        if (boost_feature_names!=[]):
            self.feature_names_[-1:-len(boost_feature_names)-1:-1] = boost_feature_names[::-1]

    def add_est(self,estimator):
        if estimator!=None:
            self.nodes.append(estimator)

    def init_layer(self, estimator_configs: dict, random_state):
        for index in range(len(estimator_configs)):
            config = estimator_configs[index].copy()
            node = NodeClassifier(
                self.layer_id, self.view_id, index, config, 
                random_state=random_state, logger_path=self.logger_path
                )
            self.add_est(node)
        return self

    def fit(self, x_train: arr, y_train: arr, 
            n_feature_origin:int, n_sample_origin:int,
            sample_weight:arr=None):
        """
        
        Returns
        -------
        y_proba_list: ndarray, shape=(#node, #sample, #class).
        wrapper_opinion_list: ndarray, shape=(#node, #sample, #class + 1)
        """
        assert len(self.nodes)>0, "在fit之前, 请先初始化: init_layer()"
        n_class = len(np.unique(y_train))
        n_sample = len(y_train)
        # y_proba_layer = np.empty((x_train.shape[0], 0))
        y_proba_list = []
        wrapper_opinion_list = []
        evidence_list = []
        # boost_features = np.empty((n_sample, 0))
        for node in self.nodes:
            y_proba, wrapper_opinion, evidence = node.fit(
                x_train, y_train, 
                n_feature_origin=n_feature_origin, 
                sample_weight=sample_weight,
                n_sample_origin=n_sample_origin,
            )
            # y_proba_layer = np.hstack(y_proba_layer, y_proba)
            y_proba_list.append(y_proba)
            wrapper_opinion_list.append(wrapper_opinion)
            evidence_list.append(evidence)
        self.feature_importances_ = self.get_importance()
        return y_proba_list, wrapper_opinion_list, evidence_list

    def predict_opinion(self,x):
        """ 返回每个结点的预测结果
        
        Returns
        -------
        proba_list: ndarray, shape = (#nodes, #samples, #classes)
        opinion_list: ndarray, shape = (#nodes, #samples, #classes+1)
        evidence_list: ndarray or None, shape = (#nodes, #samples, #classes)

        """
        opinion_list = []
        proba_list = []
        evidence_list = []
        for node in self.nodes:
            proba, opinion, evidence = node.predict_opinion(x)
            opinion_list.append(opinion)
            proba_list.append(proba)
            evidence_list.append(evidence)

        # proba = np.hstack(proba_list)
        # layer_opinion = np.hstack(opinion_list)
        if evidence is None:
            evidence_list = None
        return proba_list, opinion_list, evidence_list
    
    def _predict_opinion_final_layer(self, x_test, view_opinion_generation_method):
        """返回最后一层的预测结果, 联合的观点和概率
        Parameters
        ----------
        view_opinion_generation_method: str, 'joint' or 'mean' or 'sum'
        Returns
        -------
        proba: ndarray, shape = (#samples, #classes)
        layer_opinion: ndarray, shape = (#samples, #classes+1)
        """

        opinion_list = []
        proba_list = []
        evidence_list = []
        for node in self.nodes:
            proba, opinion, evidence = node.predict_opinion(x_test)
            opinion_list.append(opinion)
            proba_list.append(proba)
            evidence_list.append(evidence)
        if view_opinion_generation_method == "joint":
            layer_opinion = joint_multi_opinion(opinion_list)
        elif view_opinion_generation_method == "mean":
            layer_opinion = np.mean(opinion_list, axis=0)
        elif view_opinion_generation_method == "sum":
            layer_opinion = np.sum(opinion_list, axis=0)
            layer_opinion /= np.sum(layer_opinion, axis=1, keepdims=True)
        else:
            raise Exception("please set true parameter for \'view_opinion_generation_method\'")
            
        proba = layer_opinion[:, :-1].copy()
        proba /= np.sum(proba, axis=1)[:, np.newaxis]

        if (evidence is not None):
            layer_evidence = np.mean(evidence_list, axis=0)
        else:
            layer_evidence = None
        return proba, layer_opinion, layer_evidence
    
    def get_importance(self):
        """ 计算layer的特征重要性 """
        importance = 0
        n = 0
        for node in self.nodes:
            for forest in node.estimators_.values():
                importance += forest.feature_importances_
                n += 1
        importance /= n
        return importance