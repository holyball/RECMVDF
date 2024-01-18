from .base import BaseLearner
from .veiw import View
from .util_processpoolexecutor import *
from typing import Dict
import numpy as np


class Layer(BaseLearner):
    def __init__(self, layer_id, n_view, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.layer_id: int = layer_id
        self.n_view: int = n_view

        self._views = []
        
        self._is_fitted = False
        
        self._init_views(self.layer_id, **kwargs)
        pass
    
    def _init_views(self, layer_id, **kwargs):
        """初始化views学习器

        Args:
            layer_id (int): layer id
        """
        forest_params_list = kwargs.get("forest_params_list", None)
        for i in range(self.n_view):
            self._views.append(View(layer_id, i, forest_params_list, **kwargs))
    
    def fit(self, x_list, y_list, group_id=None, **kwargs):
        """训练一层, 传入的训练数据有两种形式, x_y_list或x_list+y
        
        Args:
            x_y_list (List):
            
            x_list (List):
            y (ndarray): 
            
            group_id: 验证集的组号
        """
        # if kwargs.get("x_y_list") is not None:
        #     x_y_list = kwargs.get("x_y_list")
        # elif kwargs.get("x_list") is not None:
        #     x_list = kwargs.get("x_list")
        #     y = kwargs.get("y")
        #     x_y_list = [(x, y) for x in x_list] 
        
        x_y_list = [(x, y) for x,y in zip(x_list, y_list)]     
        self._fit_layer(x_y_list, group_id=group_id, **kwargs)
    
    def _fit_layer(self, x_y_list, group_id=None, **kwargs):
        self._features_dims = { i:x.shape[1] for i, (x, _) in enumerate(x_y_list) }
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_view):
                futures.append(
                    executor.submit(fit_module, self._views[i], *x_y_list[i], group_id[i]) )
            fitted_views = [future.result() for future in futures]
        self._views = fitted_views
        self._is_fitted = True
    
    def evaluate(self, x_list, y_list, n_origin_sample, eval_func, eval_masks=None,
                 return_proba=False,
                 return_proba_mat=False):
        if eval_masks is None:
            eval_masks = [np.ones(n_origin_sample, dtype=bool) for _ in x_list]
        eval_masks = np.all(eval_masks, axis=0)
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_view):
                futures.append(
                    executor.submit(
                        evaluate_proba,self._views[i],
                        x_list[i], y_list[i], eval_func) )
            y_proba_view_mat = [future.result() for future in futures]
        
        
        y_proba_view_mat = [proba[:n_origin_sample] for proba in y_proba_view_mat]
        # ave
        y_proba_layer = np.mean(y_proba_view_mat, axis=0)
        y_origin = y_list[0][:n_origin_sample]
        
        if return_proba:
            return y_proba_layer
        if return_proba_mat:
            return y_proba_view_mat
        return eval_func(y_origin[eval_masks], y_proba_layer[eval_masks])
    
    def generate_boosting_features(self, x_list, group_id=None):
        return self._predict_proba(x_list, group_id)
    
    def _predict_proba(self, x_list, group_id_list=None):
        """_summary_

        Args:
            x_list (_type_): _description_
            group_id_list (_type_, optional): _description_. Defaults to None.

        Returns:
            ndarray: shape (n_views, n_sample, n_class)
        """
        assert self._is_fitted, "This layer not fitted"

        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_view):
                futures.append( 
                    executor.submit(generate_boosting_features_func, 
                                    self._views[i], x_list[i], group_id_list[i]) )
            y_proba_views = [future.result() for future in futures]
        return y_proba_views
    
    def predict_proba(self, x_list, y=None, return_views_proba=False):
        """预测x, 返回概率

        Args:
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
        Returns:
            ndarray
        """

        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_view):
                futures.append(
                    executor.submit(predict_module, self._views[i], x_list[i], y) )
            y_proba_view_mat = [future.result() for future in futures]
        
        # ave
        y_proba_layer = np.mean(y_proba_view_mat, axis=0)
        
        if return_views_proba:
            return y_proba_view_mat
        return y_proba_layer
    
    def predict(self, x_list, y=None):
        y_proba = self.predict_proba(self, x_list, y)
        return y_proba