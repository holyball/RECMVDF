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
from .evaluation import accuracy, f1_macro, mse_loss, aupr, auroc
from .base import BaseLearner
from .data_augmentation import assign_tags, resample, get_confidence
from copy import deepcopy
from typing import List, Tuple, Any, Dict, Union
from .util_processpoolexecutor import *
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
        self.is_stacking_for_boost_features = kwargs.get("is_stacking_for_boost_features", 
                                                         False)  # 是否累加概率增强特征

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
        self.eval_not_easy = kwargs.get("eval_not_easy", False)
        self._confidence_thresholds = []  # shape (n_layer, n_view)
        
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
        proba_list = [] # shape (span, n_view, n_sample, n_class)
        tag_list_record = [] # shape (span, n_view, n_sample)
        # 每一层的划分都保持相同
        group_id = self._split_train_data(x_list, y)
        group_id_train = [group_id for _ in range(self.n_view)]
        x_list_train = x_list
        y_list_train = [y for _ in range(self.n_view)]
        
        while layer_id < self.max_layers:
            print(f"**************** Train layer {layer_id} ****************")
            bypass_weights = []    # bypass加权权重
            
            # Weighting Bootstrap
            # if self.is_weight_bootstrap and weight_bais is not None:
            #     sample_weight[:n_origin_sample] += weight_bais
            
            # 
            if not self.accumulation:
                x_list_train = [x[:n_origin_sample] for x in x_list_train]
                y_list_train = [y[:n_origin_sample] for y in y_list_train]
                group_id_train = [gid[:n_origin_sample] for gid in group_id_train]
                
            # Adding Boosting Features
            x_list_train = self._generate_next_layer_data(layer_id, x_list_train, 
                                                          group_id_train)

            # 采样
            if (layer_id >= self.span):
                print("proba list shape: ", np.shape(proba_list))
                tag_list = self._assign_tags_train(
                    np.transpose(proba_list[-self.span:], (1,0,2,3)), 
                    y )
                if tag_list_record != []:
                    tag_list = self._tune_tags(tag_list_record, tag_list)
                print("hard and outlier number:", [np.sum((tag==3) | (tag==4)) for tag in tag_list])
                tag_list_record.append(tag_list)
                # # tag_list to csv
                # for view_id, tag in enumerate(tag_list):
                #     if not os.path.exists(f"tags_view_{view_id}.csv"):
                #         df = pd.DataFrame()
                #     else:
                #         df = pd.read_csv(f"tags_view_{view_id}.csv", index_col=0)
                #     df[f"layer_{layer_id}"] = tag
                #     df.to_csv(f"tags_view_{view_id}.csv")
                
                new_sample = self._resample(x_list, y, tag_list)    # 每个view采样的样本量不相同
                # new_sample = check_input(new_sample)

                x_list_new = [x_new for x_new, _ in new_sample]
                # 为采样的新样本增加增强特征
                x_list_new = self._generate_next_layer_data_new_sample(layer_id, 
                                                                       x_list_new )
                x_list_train = [np.vstack((x, x_new)) 
                                for x, x_new in zip(x_list_train, x_list_new)]
                y_list_train = [np.hstack((y, y_new)) 
                                for y, (_, y_new) in zip(y_list_train, new_sample)]
                
                x_new_num = [len(x) for x in x_list_new]
                
                # 更新group_id
                def update_group_id(i):
                    gid = np.hstack( (group_id_train[i], -np.ones(x_new_num[i])) )
                    return gid.astype(int)
                group_id_train = [update_group_id(i) for i in range(self.n_view)]
                for ids in group_id_train:
                    print("len(group_id_train): ", len(ids))
            
            layer = Layer(layer_id, self.n_view)

            layer.fit(x_list_train, y_list_train, group_id=group_id_train)
            
            probas = layer.evaluate(x_list_train, y_list_train, n_origin_sample, 
                                    eval_func=self.eval_func, return_proba_mat=True) 
            proba_list.append(probas)
            # 记录置信度阈值
            self._record_confidence(layer_id, probas)
            
            # 每个view的难分易分样本肯定不一样, 在这里使用记录数据划分每个view的样本类型
            # 然后由各个view进行各自的采样: 调用公共的采样函数, 生成不同view的新样本
            if layer_id >= self.span and self.eval_not_easy:
                eval_masks = [(tag != 1) | (tag != 2) for tag in tag_list]
            else:
                eval_masks = None
            eval_score = layer.evaluate(x_list_train, y_list_train, n_origin_sample,
                                        self.eval_func, eval_masks=eval_masks)

            if (eval_score > best_score):
                best_layer_id = layer_id
                best_score = eval_score
                
            
            print(f"{self.eval_func.__name__}: {eval_score:.2f}")
            n_eval_sample = n_origin_sample if eval_masks is None else np.sum(eval_masks)
            print(f"evaluation base sample number: {n_eval_sample}")
            
            if (layer_id>=self.span) and layer_id-best_layer_id>=self.early_stop_rounds:
                self._layers = self._layers[:best_layer_id+1]
                break
            self._layers.append(layer)

            layer_id += 1
            
        print(f"best layer: {best_layer_id}, best score: {best_score:.2f}")
        
    def _record_confidence(self, layer_id, proba_list):
        """_summary_

        Args:
            proba_list (ndarray): shape (n_views, n_sample, n_class)
        """
        confidence_list = get_confidence(proba_list).T
        self._confidence_thresholds.append(np.mean(confidence_list, axis=1))
    
    def _generate_next_layer_data(self, layer_id, x_list, group_id_list=None):
        """增加增强特征

        Args:
            layer_id (_type_): _description_
            x_list (_type_): _description_
            group_id (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if (layer_id==0):
            return x_list
        boosting_features = self._layers[layer_id-1].generate_boosting_features(
            x_list, group_id_list )
        
        if self.is_stacking_for_boost_features:
            x_list_new = [
                np.hstack([x, boost]) 
                for x, boost in zip(x_list, boosting_features)
            ]
        else:
            x_list_new = [ 
                np.hstack( (x_list[i][:, :self._n_origin_features[i]], 
                            boosting_features[i] ) ) 
                for i in range(self.n_view)
            ]
        return x_list_new
    
    def _generate_next_layer_data_new_sample(self, layer_id, x_list):
        """增加增强特征

        Args:
            layer_id (_type_): _description_
            x_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_list_new = x_list
        for i in range(1, layer_id+1):
            group_id_list = [-np.ones(len(x)) for x in x_list_new]
            x_list_new = self._generate_next_layer_data(i, x_list_new, group_id_list)
        
        # for i, x in enumerate(x_list_new):
        #     pd.DataFrame(x).to_csv(f"view_{i}_x_new.csv")
        return x_list_new
    
    def _split_train_data(self, x_list, y):
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
            n_splits=self.n_fold, shuffle=True, random_state=self.random_state)
        # if self.random_state is not None:
        split_view_id = np.random.randint(0, self.n_view)
        gen = skf.split(x_list[split_view_id], y)
        val_idx_fold = np.empty(y.shape[0])
        for fold_idx, (_, val_idx) in enumerate(gen):
            val_idx_fold[val_idx] = fold_idx
        return val_idx_fold
    
    def _assign_tags_train(self, proba_list, y_true):
        """分配tags

        Args:
            proba_list (ndarray): shape (n_view, span, n_samples, n_class)
            y_true (_type_): _description_
        Returns:
            ndarray: shape (n_view, n_samples), tags of all views
        """
        tag_list = []
        for single_view_mul_layer_probas in proba_list:
            tags, _ = assign_tags(single_view_mul_layer_probas, 
                                  y_true=y_true)
            tag_list.append(tags)
        return np.array(tag_list)
    
    def _assign_tags_predict(self, layer_id, proba_list):
        
        pass
    
    def  _tune_tags(self, tag_list_record, tag_list):
        """修正tags, 通过前几层的tag修正本层的tag: 
            所有曾经为1的样本都不应该具有3, 4类tag

        Args:
            tag_list_record (ndarray): shape (n_layer-span, n_view, n_sample)
            new_tag_list (ndarray): shape (n_view, n_sample)
        """
        # tag_list_record = [[1,2,3,4], [1,2,3,4]]
        tag_list_record = np.transpose(tag_list_record, (1, 0, 2)) # n_view, n_layer, n_sample
        for i in range(self.n_view):
            easy = np.any(tag_list_record[i]==1, axis=0)
            tag_list[i][easy] = 2
        # pass
        return tag_list
           
    def _resample(self, x_list, y, tag_list):
        kwargs = {
            "hard_fold":1.5,   # 困难采样倍率
            "hard_resample_kind": "random", 
            "outlier_k_neighbours": self.onn,
        }
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(self.n_view):
                futures.append(
                    executor.submit(resample, x_list[i], y, tag_list[i], 
                                    self.random_state, **kwargs) )
            new_samples = [future.result() for future in futures]
            
        # from sklearn.utils.validation import check_X_y
        # for i, (x,y) in enumerate(new_samples):
        #     try:
        #         check_X_y(x,y)
        #         print(f"view {i}, new sample shape: {x.shape}")
        #     except ValueError:
        #         print("NaN")
        #         exit()
                    
        # new_samples = []
        # for i in range(self.n_view):
        #     new_samples.append(resample(x_list[i], y, tag_list[i], self.random_state,
        #                                 **kwargs))
        return new_samples
    
    def predict_proba(self, x_list, y=None):
        y_proba_mat = []
        eval_score = 0
        best_layer_id = 0
        best_score = 0
        group_id_list = [-np.ones(len(x)) for x in x_list]
        fixed_mask = np.zeros(len(x_list[0]), dtype=bool)
        for layer_id in range(len(self._layers)):
            x_list = self._generate_next_layer_data(layer_id, x_list, group_id_list)
            # y_proba_layer = self._layers[layer_id].predict_proba(x_list)
            # y_proba_mat.append(y_proba_layer)
            y_proba_views = self._layers[layer_id].predict_proba(x_list, 
                                                                 return_views_proba=True)
            if layer_id < self.span:
                y_proba_out = self._output_proba(y_proba_views)
            else:
                # 根据y_proba_views来固定预测结果
                y_proba_out, fixed_mask = self._fixed_prediction(layer_id, y_proba_out, 
                                                                 fixed_mask,
                                                     y_proba_views)
            
            # 打印预测的结果
            if y is not None:
                eval_score = self.eval_func(y, y_proba_out)
                if (eval_score > best_score):
                    best_layer_id = layer_id
                    best_score = eval_score
                print(f"**************** Predict: layer {layer_id} ****************")
                print(f"{self.eval_func.__name__}: {eval_score:.2f}")
                
                if np.sum(fixed_mask>0):
                    eval_score_fixed = self.eval_func(y[fixed_mask], 
                                                      y_proba_out[fixed_mask])
                    print(f"fixed sample number: {np.sum(fixed_mask)}")
                    print(f"{self.eval_func.__name__} on fixed samples: {eval_score_fixed:.2f}")
        print(f"best layer: {best_layer_id}, best score: {best_score:.2f}")
        return y_proba_out
    
    def _fixed_prediction(self, layer_id, y_proba_out, fixed_mask, y_proba_views) :       
        fix_mask = self._get_fixed_mask(layer_id, y_proba_views)
        fix_mask_new = fix_mask & (~fixed_mask)
        mean_probas = np.mean(y_proba_views, axis=0)
        y_proba_out[fix_mask_new] = mean_probas[fix_mask_new]
        return y_proba_out, fix_mask | fixed_mask
    
    def _get_fixed_mask(self, layer_id, y_proba_views):
        confidence = get_confidence(y_proba_views)
        high_confidence_mask = confidence > self._confidence_thresholds[layer_id]
        high_confidence_mask = np.all(high_confidence_mask, axis=1)
        high_confidence_mask = high_confidence_mask.T
        prediction = np.array([np.argmax(proba, axis=1).squeeze()
                                for proba in y_proba_views])
        co_correct_mask = np.all(prediction == prediction[0, :], axis=0)
        
        fix_mask = high_confidence_mask & co_correct_mask
        return fix_mask
        
    def _output_proba(self, y_proba_views):
        return np.mean(y_proba_views, axis=0)
    
    def get_score(self, y_proba_list, y_list, tag_list):
        # total, easy, normal, hard, outlier
        scores = []
        for view_id in range(self.n_view):
            y = y_list[view_id]
            y_proba = y_proba_list[view_id]
            tag = tag_list[view_id]
            score_view = []
            score_view.append(self.eval_func(y, y_proba))
            for i in range(1,5):
                mask = tag == i
                if np.sum(mask) == 0:
                    score_view.append("No sample")
                else:
                    score_view.append(self.eval_func(y[mask], y_proba[mask]))
            scores.append(score_view)
        return scores
    
def check_input(x_y_list):
    # from sklearn.utils.validation import check_X_y
    # for i, x in enumerate(x_list):
    #     try:
    #         check_X_y(x)
    #     except ValueError:
    #         print(i)
    x_y_list_clean = []
    for x,y in x_y_list:
        x_df = pd.DataFrame(x)
        y_series = pd.Series(y)

        # 删除 x 中包含 NaN 的行
        x_df_clean = x_df.dropna()

        # 同时保持 y 与 x 同步，只保留 x 中没有被删除的行对应的 y
        y_series_clean = y_series[x_df_clean.index]
        
        x_y_list_clean.append((x_df_clean.values, y_series_clean.values))
    return x_y_list_clean
