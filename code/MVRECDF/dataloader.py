import numpy as np
import pandas as pd
import random
from copy import deepcopy
from typing import List, Tuple, Dict
from numpy import ndarray as arr 
from sklearn.metrics import euclidean_distances
import os

class DataLoader(object):
    def __init__(self, x_dict: Dict[int, np.ndarray], y_list=None, logger_path:str=None, 
                 onn:int=5, random_state:int=None):
        """self: origin features
        """
        # self: List[np.ndarray]
        # list.__init__([])
        # self.extend(x_list)
        self.n_views = len(x_dict)
        self.n_class = len(np.unique(y_list))
        self.features = deepcopy(x_dict)        # 存储原始样本+新样本的特征
        self.labels = {v:deepcopy(y_list) for v in range(self.n_views)}
        self.origin_features = deepcopy(self.features)
        self.origin_labels = deepcopy(y_list) if y_list is not None else None
        self.n_origin_sample = self.origin_features[0].shape[0]
        self.n_origin_features = [item.shape[1] for item in self.origin_features.values()]
        self.random_state = random_state
        self.onn = onn
        self.rng = np.random.RandomState(random_state)

        self.boost_features_names = {v:[] for v in range(self.n_views)}
        
        # 创建日志文件夹
        if logger_path is None:
            self.logger_path = "./MVUGCForest_info"
        else:
            self.logger_path = logger_path
        assert os.path.exists(self.logger_path), f"logger_path: {self.logger_path} not exist! "

        self.train_log_dir = os.path.join(self.logger_path, "sample_nn_train")
        self.predict_log_dir = os.path.join(self.logger_path, "sample_nn_predict")
        if not os.path.exists(self.train_log_dir):
            os.mkdir(self.train_log_dir)
        if not os.path.exists(self.predict_log_dir):
            os.mkdir(self.predict_log_dir)
    
    def load_data(self, v: int, return_y: bool, return_origin: bool=False, origin_mask:arr=None):
        """加载训练数据"""

        if return_origin:
            features = self.origin_features[v]
            labels = self.origin_labels
        else:
            features = self.features[v]
            labels = self.labels[v]

        if origin_mask is not None:
            # origin_, new_samples = np.
            cur_mask = np.ones(len(features), dtype=bool)
            cur_mask[:self.n_origin_sample] = origin_mask
            features = features[cur_mask]
            labels = labels[cur_mask]
        if return_y:
            return features, labels
        else:
            return features
        
    def __update_features(self, v, new_features, is_stacking, base_current_sample):
        """将增强特征堆叠到训练特征上, 将改变对象的操作数据

        Parameters
        ---------
        train_mask: 
        is_stacking: if True, 直接在特征上堆叠新特征, else, 基于原始特征堆叠新特征.
        base_current_sample: if True, 直接在所有样本上堆叠新特征, else, 基于原始样本堆叠新特征

        """
        try:
            assert new_features.shape[0] == self.features[v].shape[0], "样本数量不匹配"
        except AssertionError:
            print(f"new_features.shape[0]: {new_features.shape[0]}\tself.features[v].shape[0]: {self.features[v].shape[0]}" )
            raise
        if len(np.shape(new_features)) == 1:
            new_features = new_features[:, np.newaxis]
        
        n_boost_feature_dim = new_features.shape[1]

        if base_current_sample:
            if is_stacking or (self.features[v].shape[1] == self.n_origin_features[v]) :
                self.features[v] = np.hstack([self.features[v], new_features])
            else:
                self.features[v][:, self.n_origin_features[v]:self.n_origin_features[v]+n_boost_feature_dim] = new_features
                # self.features[v] = np.hstack([self.features[v][:, self.n_origin_features[v]], new_features])
        else:
            # 如果基于原始样本
            if is_stacking:
                # 如果直接在特征上堆叠
                stacked_features = self.features[v][:self.n_origin_sample, self.n_origin_features:]    # 原始样本上已有的增强特征
                stacking_features = np.hstack([stacked_features, new_features[:self.n_origin_sample]]) # 合并原始样本的新增强特征
                self.features[v] = np.hstack([self.origin_features[v], stacking_features])
            else:
                self.features[v] = np.hstack([self.origin_features[v], new_features[:self.n_origin_sample]])
        return self.features[v]

    def update_boost_features(self, v: int, depth:int, boost_features: arr, is_stacking:bool = False, base_current_sample:bool=True):
        """添加增强特征
            将增强特征堆叠到训练特征上, 将改变对象的操作数据

        Parameters
        ---------
        train_mask: 
        is_stacking: if True, 直接在特征上堆叠新特征, else, 基于原始特征堆叠新特征.
        base_current_sample: if True, 直接在所有样本上堆叠新特征, else, 基于原始样本堆叠新特征

        """
        self.__update_features(v, boost_features, is_stacking, base_current_sample)
        self.__update_high_order_fname(v, depth, boost_features.shape[1], "boost", is_stacking)

        return self.features[v]

    def update_intersection_features(self, v: int, depth:int, new_features: arr, ftype:str,
                                     is_stacking:bool = True, base_current_sample:bool=True,):
        """添加交互特征
        
        Parameters
        ----------
        ftype: str, 'intra' or 'inter', 交互特征类型
        """
        self.__update_features(v, new_features, is_stacking, base_current_sample)
        self.__update_high_order_fname(v, depth, new_features.shape[1], ftype, is_stacking)
        return self.features[v]

    def __update_high_order_fname(self, v, depth, n_features: int, ftype:str, is_stacking:bool, ):
        """更新增强特征名
        Parameters
        ----------
        ftype: str, 'boost', 'intra', 'inter'

        """
        boost_features_id = [i for i in range(n_features)]
        fnames = [ elem for elem in self.boost_features_names[v] if ftype in elem ]
        if fnames == []:
            self.boost_features_names[v].extend(
                [f"{ftype}_{id}_layer_{depth}" for id in boost_features_id]
            )
        else:
            cur_id = int( fnames[-1].split("_")[-1] ) + 1
            fnames_new = [f"{ftype}_{id+cur_id}_layer_{depth}" for id in boost_features_id]
            if is_stacking:
                self.boost_features_names[v].extend(fnames_new)
            else:
                pass

    def get_boost_features(self, v):
        """获取增强特征
        
        Returns
        -------
        boost_features: ndarray of shape (#samples, #boost feautres)
            如果没有增强特征, 返回[]
        """
        origin_dim = self.n_origin_features[v]
        boost_features = self.features[v][:, origin_dim:]
        return boost_features
    
    def resample(self, v: int, depth: int, marks: arr, fold: float=2) -> Tuple[arr, arr]:
        """对标记的困难样本和离群样本进行采样
        
        Parameters
        ----------
        x:  
        y: 
        marks: ndarray of shape (#samples, ). 标记数组.
            "normal"-1, "easy"-2, "hard"-3, "outlier"-4
        fold: float, > 1.0, default = 2.0
            困难样本采样的倍数
        
        Returns
        -------
        x_new: ndarray of shape (#samples, #origin_features). 
            if sample number maybe is 0, then the value of shape in axis 0 is 0.
        y_new: 
        """
        assert len(marks) == self.n_origin_sample, "样本数量不匹配"
        
        self.update_rng(v+depth)  # 更新随机种子

        from sklearn.metrics import euclidean_distances
        x = self.origin_features[v]
        y = self.origin_labels
        # print(x.shape)
        normal_mask = marks == 1
        easy_mask = marks == 2
        hard_mask = marks == 3
        outlier_mask = marks == 4
        k_hard = 10 if int (len(marks) * 0.01) < 10 else int (len(marks) * 0.01)
        k_outlier = 10 if int(len(marks) * 0.01) < 10 else int(len(marks) * 0.01)
        x_new_for_hard, y_new_for_hard = np.empty(shape=(0, x.shape[1])), []
        x_new_for_outlier, y_new_for_outlier = np.empty(shape=(0, x.shape[1])), []

        # normal_sample_knn_df = inspect_sample_nn(x, y, normal_mask, 30)
        # normal_sample_knn_df.to_csv(self.train_log_dir + f"/normal_knn_layer_{depth}_view_{v}.csv")        
        # easy_sample_knn_df = inspect_sample_nn(x, y, easy_mask, 30)
        # easy_sample_knn_df.to_csv(self.train_log_dir + f"/easy_knn_layer_{depth}_view_{v}.csv")
        # hard_sample_knn_df = inspect_sample_nn(x, y, hard_mask, 30)
        # hard_sample_knn_df.to_csv(self.train_log_dir + f"/hard_knn_layer_{depth}_view_{v}.csv")
        # outlier_sample_knn_df = inspect_sample_nn(x, y, outlier_mask, 30)
        # outlier_sample_knn_df.to_csv(self.train_log_dir + f"/outlier_knn_layer_{depth}_view_{v}.csv")

        log_sample_nn(x, y, marks=marks, k=10, log_dir=self.train_log_dir, depth=depth, v=v)

        # 随机采样
        """
        for x_hard, y_hard in zip(x[hard_mask], y[hard_mask]):
            x_hard = x_hard[np.newaxis, :]
            y_hard = np.array([y_hard])
            distances = euclidean_distances(x_hard, x).squeeze()
            x_to_resample = x[idx_knn]
            y_to_resample = y[idx_knn]
            x_new, y_new = self._resample(x_to_resample, y_to_resample)
            x_new_for_hard = np.vstack([x_new_for_hard, x_new])
            y_new_for_hard = np.hstack([y_new_for_hard, y_new])
        """

        # 插值采样
        # 将hard_sample划分类别
        for c in np.unique(y):
            mask_c = (y == c) & hard_mask   # 类标签为c类 的困难样本的mask, shape(#n_sample,)
            # print(np.sum(mask_c))
            idx_not_c = np.unique(
                self.rng.choice(np.argwhere(y != c).squeeze(), int(np.sum(mask_c)*fold))
                )
            # if idx_not_c
            if np.sum(mask_c) < 6: continue
            try:
                x_to_sample = np.vstack([x[mask_c], x[idx_not_c]])
                y_to_sample = np.hstack([y[mask_c], y[idx_not_c]])
                x_new, y_new = self._resample(x_to_sample, y_to_sample, kind="smote")
                x_new_for_hard = np.vstack([x_new_for_hard, x_new])
                y_new_for_hard = np.hstack([y_new_for_hard, y_new])
            except ValueError:
                print(f"sum(mask_c): {np.sum(mask_c)}, len(idx_not_c): {len(idx_not_c)}")
        # # 将离群样本划分类别并采样
        # for c in np.unique(y):
        #     mask_c = (y == c) & outlier_mask
        #     # print(np.sum(mask_c))
        #     idx_not_c = np.unique(self.rng.choice(np.argwhere(y != c).squeeze(), np.sum(mask_c)*5))
        #     if np.sum(mask_c) < 6: continue
        #     x_to_sample = np.vstack([x[mask_c], x[idx_not_c]])
        #     y_to_sample = np.hstack([y[mask_c], y[idx_not_c]])
        #     x_new, y_new = self._resample(x_to_sample, y_to_sample, kind="smote")
        #     x_new_for_outlier = np.vstack([x_new_for_outlier, x_new])
        #     y_new_for_outlier = np.hstack([y_new_for_outlier, y_new])
        # 将离群样本进行同类插值采样
        # x_new_for_outlier, y_new_for_outlier = self.resample_outlier(x, y, outlier_mask)

        x_new_for_outlier, y_new_for_outlier = self.resample_outlier(\
            self.origin_features[v], self.origin_labels, outlier_mask)
        if x_new_for_outlier == []:
            x_new_for_outlier = np.empty(shape=(0, x.shape[1]))
        # 查看为离群样本新采样的样本的原始近邻样本情况
        x_sample_all = np.vstack([x, x_new_for_outlier])
        y_sample_all = np.hstack([y, y_new_for_outlier])
        outlier_mask_sample_all = np.zeros(x_sample_all.shape[0], dtype=bool)
        outlier_mask_sample_all[np.argwhere(outlier_mask).squeeze()] = True
        outlier_new_sample_knn_df = inspect_sample_nn(\
            x_sample_all, y_sample_all, outlier_mask_sample_all, 30)
        outlier_new_sample_knn_df.to_csv(\
            self.train_log_dir + f"/resample_outlier_knn_layer_{depth}_view_{v}.csv")
        # 堆叠离群样本的新样本
        x_new = np.vstack([x_new_for_hard, x_new_for_outlier])
        y_new = np.hstack([y_new_for_hard, y_new_for_outlier])

        # # 更新features, labels
        # self.update_x_y(x_new, y_new, v)
        
        # 让x_new至少是2维矩阵
        x_new = np.atleast_2d(x_new)

        return x_new, y_new
            
    def resample_layer(self, depth: int, marks:arr, fold: float=2) -> Tuple[Dict[int, arr], arr]:
        """层采样, 对所有view的原始直接合并, 
           直接在合并后的数据集上对困难样本与离群样本进行采样
        Parameters
        ----------
        x:  
        y: 
        marks: ndarray of shape (#samples, ). 标记数组.
            "normal"-1, "easy"-2, "hard"-3, "outlier"-4
        fold: float, > 1.0, default = 2.0
            困难样本采样的倍数
        
        Returns
        -------
        x_new: ndarray of shape (#samples, #origin_features). 
            if sample number maybe is 0, then the value of shape in axis 0 is 0.
        y_new:
        """
        assert len(marks) == self.n_origin_sample, "样本数量不匹配"

        self.update_rng(depth)  # 更新随机种子

        x = np.hstack(self.origin_features.values())
        y = self.origin_labels
        normal_mask = marks == 1
        easy_mask = marks == 2
        hard_mask = marks == 3
        outlier_mask = marks == 4
        k_hard = 10 if int (len(marks) * 0.01) < 10 else int (len(marks) * 0.01)
        k_outlier = 10 if int(len(marks) * 0.01) < 10 else int(len(marks) * 0.01)

        # 装新样本的容器
        x_new_for_hard, y_new_for_hard = np.empty(shape=(0, x.shape[1])), []
        x_new_for_outlier, y_new_for_outlier = np.empty(shape=(0, x.shape[1])), []

        # 困难样本的插值采样
        #     对hard_sample逐类别采样
        for c in np.unique(y):
            mask_c = (y == c) & hard_mask       # 类标签为c类 的困难样本的mask, shape(#n_sample,)
            # print(np.sum(mask_c))
            idx_not_c = np.unique(
                self.rng.choice(
                    np.argwhere(y != c).squeeze(), 
                    int(np.sum(mask_c)*fold)
                )
            )        # 类标签不为c类的困难样本索引
            if np.sum(mask_c) < 6: continue     # 如果类别对应的困难样本量小于6, 则不采样
            try:
                x_to_sample = np.vstack([x[mask_c], x[idx_not_c]])
                y_to_sample = np.hstack([y[mask_c], y[idx_not_c]])
                x_new, y_new = self._resample(x_to_sample, y_to_sample, kind="smote") # "smote", "borderlinesmote"
                x_new_for_hard = np.vstack([x_new_for_hard, x_new])
                y_new_for_hard = np.hstack([y_new_for_hard, y_new])
            except ValueError:
                print(f"sum(mask_c): {np.sum(mask_c)}, len(idx_not_c): {len(idx_not_c)}")

        # outlier(less_info)样本的插值采样
        x_new_for_outlier, y_new_for_outlier = self.resample_outlier(
            x, 
            self.origin_labels, 
            outlier_mask, 
        )
        if x_new_for_outlier == []:
            x_new_for_outlier = np.empty(shape=(0, x.shape[1]))

        # 堆叠离群样本的新样本
        x_new = np.vstack([x_new_for_hard, x_new_for_outlier])
        y_new = np.hstack([y_new_for_hard, y_new_for_outlier])

        # 让x_new至少是2维矩阵
        x_new = np.atleast_2d(x_new)

        # 用字典存储新样本的特征
        x_new_dict = {}
        counts = 0
        for v, n_feature in enumerate(self.n_origin_features):
            x_new_dict[v] = x_new[:, counts : counts + n_feature]
            counts += n_feature

        return x_new_dict, y_new, (len(x_new_for_hard), len(x_new_for_outlier))
    
    def resample_layer_for_views(
        self, 
        depth: int, 
        marks:np.ndarray, 
        fold: float=2.,
        add_noise: bool=False,
        mu:float=0,
        sigma: float=.5,
    ) -> Tuple[Dict[int, arr], arr]:
        """层采样, 但是为每个view单独采样
        
        Parameters
        ----------
        depth: 当前的深度
        marks: 样本的标记
        fold: 采样倍率
        add_noise: 是否添加噪声
        mu: 噪声均值
        sigma: 噪声方差
        """
        assert len(marks) == self.n_origin_sample, "样本数量不匹配"
        
        self.update_rng(depth)  # 更新随机种子

        hard_mask = marks == 3
        outlier_mask = marks == 4
        x_new_for_hard = {f_n: np.empty(shape=(0, f.shape[1])) for f_n, f in self.origin_features.items()}
        x_new_for_outlier = {f_n: np.empty(shape=(0, f.shape[1])) for f_n, f in self.origin_features.items()}
        y_new_for_hard, y_new_for_outlier = [], []
        for c in np.unique(self.origin_labels):
            mask_c = ( self.origin_labels==c ) & hard_mask       # 类标签为c类 的困难样本的mask, shape
            idx_not_c = np.unique(
                self.rng.choice(
                    np.argwhere(self.origin_labels != c).squeeze(),
                    int(np.sum(mask_c)*fold)
                )
            )        # 类标签不为c类的困难样本索引
            if np.sum(mask_c) < 6:
                continue     # 如果类别对应的困难样本量小于6, 则不采样
            for f_name, x in self.origin_features.items():
                try:
                    x_to_sample = np.vstack([x[mask_c], x[idx_not_c]])
                    y_to_sample = np.hstack(
                        [self.origin_labels[mask_c], 
                         self.origin_labels[idx_not_c]]
                    )
                    x_new, y_new = self._resample(
                        x_to_sample, y_to_sample, kind="smote"
                    )
                    x_new = np.atleast_2d(x_new)

                    # 添加噪声
                    if add_noise:
                        n_added, n_features = x_new.shape
                        noise = sigma * self.rng.randn(n_added, n_features) + mu
                        x_new = x_new + noise

                    x_new_for_hard[f_name] = np.vstack( (x_new_for_hard[f_name], x_new) ) 
                except ValueError:
                    print(
                        f"sum(mask_c): {np.sum(mask_c)}, len(idx_not_c): {len(idx_not_c)}")
            y_new_for_hard = np.hstack( ( y_new_for_hard, y_new ) )
        
        # outlier(less_info)样本的插值采样
        x_new_for_outlier, y_new_for_outlier = self.resample_outlier_dict(
            self.origin_features, self.origin_labels, outlier_mask
        )
        
        x_new_dict = {}
        for f_name in self.origin_features.keys():
            x_new_dict[f_name] = np.vstack( [ x_new_for_hard[f_name], 
                                              x_new_for_outlier[f_name] ]
            )
        y_new = np.hstack( [y_new_for_hard, 
                            y_new_for_outlier ] 
        )

        return x_new_dict, y_new, len(y_new_for_hard), len(y_new_for_outlier)
    
    def update_samples_x_y(self, x_new, y_new, v, is_accumulation:bool, n_train_origin):
        """更新self.features和self.labels
        
        Parameters
        ----------
        is_accumulation: bool, if True: 直接进行累加; if False: 基于原始样本累加

        n_train_origin: int, 传入训练的原始样本数量
        """
        if len(x_new) == 0: 
            return True
        try:
            if is_accumulation:
                
                    self.features[v] = np.vstack([self.features[v], x_new])
                    self.labels[v] = np.hstack([self.labels[v], y_new])
            else:
                self.features[v] = np.vstack([ self.features[v][:n_train_origin], x_new ])
                self.labels[v] = np.hstack([ self.labels[v][:n_train_origin], y_new ])
        except Exception:
            print("update failed")
            return False
        else:
            return True

    def _resample(self, x, y, kind='random'):
        """重采样
        
        Parameters
        ----------
        x: ndarray, shape of (#samples, #features)
        y: ndarray, shape of (#samples, )
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
            if self.random_state:
                sm.set_params(random_state=self.random_state)
            x_res, y_res = sm.fit_resample(x, y)
            x_new, y_new = x_res[len(x):], y_res[len(x):]

            return x_new, y_new
        except ValueError:
            print(f"困难样本采样失败啦, 困难样本的分布为: {np.unique(y, return_counts=True)}")
            # print(np.unique(y, return_counts=True)[1])
            return np.empty((0, x.shape[1])), np.empty(0)
    
    def resample_outlier(self, x: np.ndarray, y: np.ndarray, 
                         mask: np.ndarray, k: int=None) -> Tuple[np.ndarray, np.ndarray]:
        """离群样本的线性插值采样
        最近邻 不包括 特征相同的样本
        Paramters
        ---------
        y: ndarray of shape (#samples, ),   
            传入的样本可能是origin sample, 也可能是origin sample + manual sample
        mask: ndarray of shape (#origin sample, ) dtype = bool, 将离群样本标记出来
        k: int, 希望的同类最近邻样本数量

        """
        x_new = []
        y_new = []

        k = k if k is not None else self.onn
        
        ids = np.argwhere(mask).reshape(-1)
        for i in ids:
            instance = x[i]
            instance_label = y[i]
            
            # 计算 instance 的 k 近邻样本索引
            distances = np.linalg.norm(x - instance, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]  # 不包括自身
            
            # 统计 k 近邻样本中的异类样本数量
            n_opposite = np.sum(y[nearest_indices] != instance_label)
            
            if n_opposite > 0:
                # 计算采样步长的上限
                step_upper = distances[nearest_indices[n_opposite-1]]
                
                if step_upper < 1e-3 or step_upper == np.NaN:
                    # 如果步长上限过小或无效，不进行采样
                    continue
                
                # 生成新样本
                steps = self.rng.uniform(1e-3, step_upper, n_opposite).reshape((-1, 1))
                diffs = x[nearest_indices[:n_opposite]] - instance
                x_res = np.atleast_2d(instance) + steps * diffs
                y_res = np.full(n_opposite, instance_label)
                
                x_new.append(x_res)
                y_new.append(y_res)
        
        if len(x_new) > 0:
            x_new = np.vstack(x_new)
            y_new = np.hstack(y_new)
        
        return x_new, y_new

    def resample_outlier_dict(self, x_dict: Dict, y: np.ndarray, mask:np.ndarray, k:int=None ):
        """对字典数据进行采样"""
        k = k if k is not None else self.onn

        x = np.hstack(x_dict.values())
        ids = np.argwhere(mask).reshape(-1)
        x_new_dict = {f_n: np.empty(shape=(0, f.shape[1])) for f_n, f in x_dict.items()}
        y_new = []
        for i in ids:
            instance = x[i]
            instance_label = y[i]

            # 计算 instance 的 k 近邻样本索引
            distances = np.linalg.norm(x - instance, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]  # 不包括自身
            # 统计 k 近邻样本中的异类样本数量
            n_opposite = np.sum(y[nearest_indices] != instance_label)
            if n_opposite > 0:

                # 生成新样本
                for f_name, x_sub in x_dict.items():

                    steps = self.rng.uniform(1e-3, 1, n_opposite).reshape((-1, 1))
                    diffs = x_sub[nearest_indices[:n_opposite]] - x_sub[i]
                    x_res = np.atleast_2d(x_sub[i]) + steps * diffs
                    x_new_dict[f_name] = np.vstack( (x_new_dict[f_name], x_res) )
                    
                    # # 采样前 x_sub[i] 的近邻样本
                    # print(f"x_sub[{i}]: {y[i]}")
                    # d = euclidean_distances(np.atleast_2d(x_sub[i]), x_sub)[0]
                    # idx_tmp = np.argsort(d)[1:10]
                    # print(f"neighbour labels: {y[idx_tmp]}, idx: {idx_tmp}")
                    # # 采样后 x_sub[i] 的近邻样本
                    # x_add = np.vstack([x_sub, x_res])
                    # y_add = np.hstack( [y, np.full(n_opposite, instance_label)] )
                    # d2 = euclidean_distances(np.atleast_2d(x_sub[i]), x_add)[0]
                    # idx_2 = np.argsort(d2)[1:10]
                    # print(f"neighbour labels: {y_add[idx_2]}, idx: {idx_2}")

                y_res = np.full(n_opposite, instance_label)
                y_new = np.hstack( (y_new, y_res) )
        return x_new_dict, y_new
                    
    def update_rng(self, depth:int):
        if self.random_state is not None:
            self.random_state += 666*depth
            self.rng = np.random.RandomState(self.random_state)


def inspect_sample_nn(x, y, mask, k):
    """查看感兴趣样本的近邻样本的标签
    
    Parameters:
    ----------
    mask: ndarray of shape (#samples, ), 
        the True point represents the interesting point
    k: int, 近邻样本的数量
    
    Returns:
    -------
    pd.DataFrame 
    """
    if np.sum(mask) == 0:
        return pd.DataFrame([None])


    dist_mat = euclidean_distances(x[mask], x)
    labels = []
    distances = []
    for line in dist_mat:
        idx_knn = np.argsort(line)[:k]
        labels.append(y[idx_knn])
        distances.append(line[idx_knn])
    index = np.atleast_1d(np.argwhere(mask).squeeze())
    labels_df = pd.DataFrame(labels, index=index)
    distances_df = pd.DataFrame(distances, index=index)
    return pd.concat([labels_df, distances_df], axis=1)

def log_sample_nn(x, y, marks, k, log_dir: str, depth: int, v: int):
    """把四类样本的近邻结果缓存

    """
    normal_mask = marks == 1
    easy_mask = marks == 2
    hard_mask = marks == 3
    outlier_mask = marks == 4

    normal_sample_knn_df = inspect_sample_nn(x, y, normal_mask, k)
    normal_sample_knn_df.to_csv(log_dir + f"/normal_knn_layer_{depth}_view_{v}.csv")        
    easy_sample_knn_df = inspect_sample_nn(x, y, easy_mask, k)
    easy_sample_knn_df.to_csv(log_dir + f"/easy_knn_layer_{depth}_view_{v}.csv")
    hard_sample_knn_df = inspect_sample_nn(x, y, hard_mask, k)
    hard_sample_knn_df.to_csv(log_dir + f"/hard_knn_layer_{depth}_view_{v}.csv")
    outlier_sample_knn_df = inspect_sample_nn(x, y, outlier_mask, k)
    outlier_sample_knn_df.to_csv(log_dir + f"/outlier_knn_layer_{depth}_view_{v}.csv")

        
if __name__ == "__main__":  
    pass