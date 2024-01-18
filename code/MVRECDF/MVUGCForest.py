'''           
Author: tanqiong
Date: 2023-04-18 17:09:34
LastEditTime: 2023-10-09 18:17:06
LastEditors: tanqiong
'''
import numpy as np
import pandas as pd
from numpy import ndarray as arr
from .logger import get_logger, get_opinion_logger, get_custom_logger, mkdir
from .veiw import View
from .layer import Layer
from .node import NodeClassifier
from .uncertainty import joint_multi_opinion, opinion_to_proba
from .util import (record_opinion_res, df_to_csv, 
                    get_stage_matrix, get_scores, metrics_name)
from .dataloader import DataLoader
from .evaluation import accuracy, f1_macro, mse_loss, aupr, auroc

from copy import deepcopy
from typing import List, Tuple, Any, Dict, Union
import os
import warnings
warnings.filterwarnings("ignore")
# 为了保证可以完整打印opinion
np.set_printoptions(threshold=np.inf)
# from ..util import save_opinions

class MVUGCForest(object):

    def __init__(self, config: dict):
        self.random_state = config.get("random_state", None)

        # boost features, 
        self.is_stacking_for_boost_features = config.get("is_stacking_for_boost_features", False)  # 是否累加概率增强特征
        self.boost_feature_type: str = config.get("boost_feature_type", 'probability')
        self.replace_belief: bool = config.get("replace_belief", False)                     # 是否使用proba替换opinion中的belief

        # cluster samples
        self.cluster_samples_layer = config.get("cluster_samples_layer", True)
        self.cluster_method: Union[str, bool] = config.get("cluster_method", 'uncertainty')     # 'uncertainty', 'span', 'uncertainty_bins'
        self.n_bins:int = config.get("n_bins", 30)     # use for cluster_method = 'uncertianty_bins'
        self.ta:float = config.get("ta", 0.7)           # DBC的阈值参数
        self.alpha: float = config.get("alpha", 0.7)    # gcForest_cs的阈值参数

        # resample
        self.is_resample = config.get("is_resample", True)
        self.onn = config.get("onn", 3)     # outlier nearest neighbour
        self.layer_resample_method = config.get("layer_resample_method", "integration")   # "integration", "respective"
        self.accumulation = config.get("accumulation", False)  # 人工样本是否累加

        # uncertainty
        self.evidence_type = config.get("evidence_type", "probability")   
        self.uncertainty_basis = config.get("uncertainty_basis", "no_uncertainty")

        # training
        self.span = config.get("span", 2)
        self.max_layers = config.get("max_layers", 10)
        self.early_stop_rounds = config.get("early_stop_rounds", 2)
        self.is_weight_bootstrap = config.get("is_weight_bootstrap", True) # 加权 bootstrap
        self.is_save_model = config.get("is_save_model")
        self.train_evaluation = config.get("train_evaluation", accuracy)
        self.view_opinion_generation_method = config.get("view_opinion_generation_method", 'joint') # 'joint', 'mean'

        # prediction
        self.bypass_threshold = config.get("bypass_threshold", 0.6)
        self.is_des = config.get("is_des", False)
        self.use_layerwise_opinion = config.get("use_layerwise_opinion", True)              # 是否使用所有层的联合opinion
        # 如果设置了bypass参数, 则将它注册到类属性中
        if config.get("bypass_prediction", None) is not None:
            self.bypass_prediction = config.get("bypass_prediction", True)

        # node configs
        self.n_fold = config.get("n_fold", 5)

        # 增加node_config
        self.estimator_configs = config.get("estimator_configs").copy()
        for item in self.estimator_configs:        # Dynamic Ensemble Selection
            item["evidence_type"] = self.evidence_type
            item["uncertainty_basis"] = self.uncertainty_basis

        # some variables for training or prediction
        self.n_view: int
        self.n_class: int
        self.views: Dict[int, View]             # views
        self.bypass_weights = []

        # 记录中间结果的变量
        self.best_layer_id: int # 最佳层的层号
        self.layers_opinions_list_train = []
        self.layers_opinions_list_predict = []
        self.layer_view_opinion_dict_predict = {}

        self.uncertainty_threshold = {}

        # RIT variable
        self.new_features_lst = {}
        self.new_feature_stability = {} 
        self.bootstrap_feature_importance = {}
        
        self.intra_selected_idx = {}
        self.inter_selected_idx = {}        # 
        self.inter_operations_dict = {}      # 存储模态间交互操作 operations的字典
        self.inter_orders_dict = {}         # 存储模态间交互顺序的字典
        self.inter_remove_idx_dict = {}     # 用于删除高相关性的高阶交互特征对

        # init logger
        logger_path:str = config.get("logger_path", "MVUGCForest_info")
        if logger_path.find("MVUGCForest_info") == -1:
            logger_path = os.path.join(logger_path, "MVUGCForest_info")
        if not os.path.exists(logger_path):
            try: 
                os.mkdir(logger_path)
            except Exception: 
                raise Exception("路径错误")
        self.register_logger(logger_path)
        # 记录参数配置
        self.LOGGER_MULTIVIEW_T.info(str(config))
        self.LOGGER_MULTIVIEW_V.info(str(config))
        self.logger_path = logger_path
        
    def register_logger(self, path: str):
        """注册LOGGER"""
        if "MVUGCForest_info" not in path:
            path = os.path.join(path, "MVUGCForest_info")
        if not os.path.exists(path):
            try: 
                os.mkdir(path)
            except Exception: 
                raise Exception("路径错误")
        self.OPINION_LOGGER_T = get_custom_logger(
            "opinion_train", "opinions_train.txt", path)
        self.OPINION_LOGGER_V = get_custom_logger(
            "opinion_predict", "opinions_test.txt", path)
        self.LOGGER_MULTIVIEW_T = get_custom_logger(
            "multiview_train", "multiview_fit.txt", path)
        self.LOGGER_MULTIVIEW_V = get_custom_logger(
            "multiview_predict", "multiview_predict.txt", path)

    def fit_multiviews(self, x_train: List[arr], y_train: arr, 
                       evaluate_set: str="all"):
        """fit for multi-views data

        Parameters
        ----------
        x_train: ndarray, shape=(#views, #samples, #features)
        y_train: ndarray, shape=(#samples).
        evaluate_set: str, default "all", or "trouble" 
            评估性能时, 是否要排除易份样本, 默认不排除易分样本("all")
        """
        # some supporting variable
        self.n_view = len(x_train)
        assert self.n_view > 1, "view的数量必须大于1"

        views:Dict[int, View] = {v:View() for v in range(self.n_view)}
        x_train_dict, n_feature_list = {}, []
        y_origin = y_train.copy()
        for v in range(self.n_view):
            a, b, _ = self.preprocess(x_train[v], y_train)
            x_train_dict[v] = a
            n_feature_list.append(b)
        dl_train = DataLoader(x_train_dict, y_origin, logger_path = self.logger_path, 
                              onn=self.onn,
                              random_state=self.random_state)
        n_class = dl_train.n_class
        self.n_class = n_class
        n_origin_sample = dl_train.n_origin_sample
        n_train_origin_sample = dl_train.n_origin_sample
        evaluate = self.train_evaluation
        evaluate_name:str = evaluate.__name__
        best_layer_id = 0
        depth = 0
        best_layer_evaluation = -float('inf') if evaluate_name.find("loss") == -1 else float('inf')
        views_opinions_origin_dict = {v:[] for v in range(self.n_view)} # 每个view的opinions, shape of (#views, #layers,) 
        layers_opinions_list_origin = []                                # 每一层的view之间的联合opinions, shape of (#layers, )
        sample_marks_layer_df = pd.DataFrame()                          # 有可能第0层是没有做样本划分, 因此需要用DataFrame, 可以用索引
        train_origin_mask = np.ones(n_origin_sample, dtype=bool)
        opinion_fixed = np.empty((n_origin_sample, n_class+1))
        views_evidence_origin_dict = {v:[] for v in range(self.n_view)}

        # 样本权重
        sample_weight = np.ones(n_origin_sample)
        weight_bais = None    # 权重偏移(置信度低的样本, 权重大)

        if evaluate_name.find("loss") != -1 and self.uncertainty_basis != "evidence":
            raise Exception("使用loss作为损失函数应使用evidence作为uncertainty basis")

        # 生长级联森林
        while depth < self.max_layers:
            baypass_weights = []    # bypass加权权重

            print(f"train layer {depth}")
            # 查看本层训练使用的样本数量
            print(f"layer {depth} sample number {[len(item) for item in dl_train.features.values()]}")
            self.OPINION_LOGGER_T.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(depth))
            self.LOGGER_MULTIVIEW_T.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(depth))
            
            # 如果样本加权
            if self.is_weight_bootstrap and weight_bais is not None:
                sample_weight[:n_origin_sample] += weight_bais
            
            for v in range(self.n_view):
                print(f"  train view {v} ")
                self.LOGGER_MULTIVIEW_T.info(f"  -----------view-{v}-----------")
                
                n_train_origin_sample = np.sum(train_origin_mask)
                
                # 获取view对应的训练数据
                x_train_dict[v] = dl_train.features[v]
                y_train = dl_train.labels[v]

                # 查看训练的特征的组成, 多少个原始特征, 多少个增强特征, 多少个交互特征
                n_feature_cur_ = x_train_dict[v].shape[1]
                _, n_boost_, n_intra_, n_inter_ = self.__print_features_info(v, dl_train)
                print(f"  feature dims:{n_feature_cur_}\tn boost features: {n_boost_};\tn intra features: {n_intra_}\tn inter features: {n_inter_}")
                self.LOGGER_MULTIVIEW_T.info(f"  feature dims:{n_feature_cur_}\tn boost features: {n_boost_};\tn intra features: {n_intra_}\tn inter features: {n_inter_}")

                sample_weight_train = np.ones(len(y_train))
                sample_weight_train[:n_origin_sample] = sample_weight[:n_origin_sample]

                # 初始化layer并拟合
                layer = Layer(depth, v, self.logger_path).init_layer(
                    self.estimator_configs, self.random_state
                    )
                proba_list, opinion_list, evidence_list = layer.fit(
                    x_train_dict[v], y_train, 
                    n_feature_origin=dl_train.n_origin_features[v],
                    n_sample_origin = dl_train.n_origin_sample,
                    sample_weight = sample_weight_train,
                )
                # set feature_names 
                layer.set_feature_names_(dl_train.boost_features_names[v])

                views[v].add_layer(layer)        

                proba_train_origin_list = [proba[:n_train_origin_sample] for proba in proba_list]        # 保留原始训练样本的proba, shape=(#nodes, #training samples, #classes)
                opinion_train_origin_list = [opinion[:n_train_origin_sample] for opinion in opinion_list] # 保留原始训练样本的opinion,shape=(#nodes, #training samples, #classes+1)

                # 更新每个nodes的预测的不确定度(由置信度导出)
                for i in range(len(opinion_train_origin_list)):
                    op_tmp = opinion_train_origin_list[i]   # 修改op_tmp中的内容将直接修改 opinion_train_origin_list[i] 对应的内容
                    jingquelv = evaluate(y_origin, op_tmp[:, :-1])
                    op_tmp[:, -1] = jingquelv * np.max(op_tmp[:, :-1], axis=1)
                    op_tmp[:, -1] = 1-op_tmp[:, -1] # 将置信度替换为不确定度

                # 是否使用不确定性作为增强特征
                if self.boost_feature_type == 'probability':
                    boost_features = np.hstack(proba_list)
                else:
                    raise ValueError("Parameter error")

                # 更新第v个view的训练数据的特征
                if boost_features is not None: # 增强特征
                    x_train_dict[v] = dl_train.update_boost_features(
                        v, 
                        depth,
                        boost_features, 
                        self.is_stacking_for_boost_features,
                    )

                # 计算单个view的opinion, 基于所有原始数据
                _opinion_origin_view_i = self.generate_view_opinion(
                    opinion_train_origin_list,
                    self.view_opinion_generation_method,
                )
                # 更新train_origin_mask标记的样本
                if depth == 0:
                    opinion_origin_view_i = _opinion_origin_view_i
                else:
                    opinion_origin_view_i = deepcopy(views_opinions_origin_dict[v][-1])
                    opinion_origin_view_i[train_origin_mask] = _opinion_origin_view_i
                views_opinions_origin_dict[v].append(opinion_origin_view_i)

                # 日志记录 每个view的opinion和精确度, base: 传入训练的原始样本
                self.OPINION_LOGGER_T.info(
                    f"  view opinion,\tview: {v}, \n{opinion_origin_view_i}")
                score_view_origin_sample = evaluate(y_origin, opinion_to_proba(opinion_origin_view_i))
                
                baypass_weights.append(score_view_origin_sample)
                self.LOGGER_MULTIVIEW_T.info(
                    "  The evaluation[{}] of view_{} is {:.4f}".format(evaluate_name, v, score_view_origin_sample))
                # self.LOGGER_MULTIVIEW_T.info(
                #     "  The f1-score of view_{} is {:.4f}".format(v, f1_macro(y_origin, opinion_to_proba(opinion_origin_view_i))))
                # self.LOGGER_MULTIVIEW_T.info(
                #     "  The accuracy of view_{} is {:.4f}".format(v, accuracy(y_origin, opinion_to_proba(opinion_origin_view_i))))
                
                self.LOGGER_MULTIVIEW_T.info(f"  sample numbers of class:\t{[np.sum(y_train==label) for label in range(n_class)]}")
                self.LOGGER_MULTIVIEW_T.info(f"  feature shape:\t{np.shape(x_train_dict[v])}")
            self.bypass_weights.append(baypass_weights)

            # 计算本层多个view的联合opinion, 基于原始的样本计算
            joint_opinion_cur_layer_origin = joint_multi_opinion(
                [view_opinions[-1] for view_opinions in views_opinions_origin_dict.values()])
            
            # enable layer_opinion
            if self.use_layerwise_opinion and depth>0:
                joint_opinion_cur_layer_origin = joint_multi_opinion(
                    [ joint_opinion_cur_layer_origin, layers_opinions_list_origin[-1] ]
                )
            # 计算原始样本的uncertainty, 作为boostrap的weight
            if self.is_weight_bootstrap:
                weight_bais = 1-joint_opinion_cur_layer_origin[:, -1]
                # 对uncertainty做归一化
                min_val = np.min(weight_bais)
                max_val = np.max(weight_bais)
                weight_bais = (weight_bais - min_val) / (max_val - min_val)

            layers_opinions_list_origin.append(joint_opinion_cur_layer_origin)
            self.OPINION_LOGGER_T.info(
                f"layer opinion:\tlayer: {depth}, \n{joint_opinion_cur_layer_origin}")
            
            # 基于层的综合opinion划分难分样本
            sample_marks_layer = None       # 标记
            if self.cluster_method:
                sample_marks_layer = self.cluster_sample(self.cluster_method, 
                                                             depth, 
                                                             layers_opinions_list_origin, 
                                                             y_origin,
                )                
                # 记录训练日志
                self.LOGGER_MULTIVIEW_T.info("-----------sample clusters-----------")
                self.LOGGER_MULTIVIEW_T.info(f"  number of normal sample\t{np.sum(sample_marks_layer==1)}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 0: {np.sum((sample_marks_layer==1) & (y_origin==0))}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 1: {np.sum((sample_marks_layer==1) & (y_origin==1))}")
                self.LOGGER_MULTIVIEW_T.info(f"  number of easy sample\t{np.sum(sample_marks_layer==2)}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 0: {np.sum((sample_marks_layer==2) & (y_origin==0))}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 1: {np.sum((sample_marks_layer==2) & (y_origin==1))}")
                self.LOGGER_MULTIVIEW_T.info(f"  number of hard sample\t{np.sum(sample_marks_layer==3)}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 0: {np.sum((sample_marks_layer==3) & (y_origin==0))}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 1: {np.sum((sample_marks_layer==3) & (y_origin==1))}")
                self.LOGGER_MULTIVIEW_T.info(f"  number of outlier sample\t{np.sum(sample_marks_layer==4)}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 0: {np.sum((sample_marks_layer==4) & (y_origin==0))}")
                self.LOGGER_MULTIVIEW_T.info(f"  class 1: {np.sum((sample_marks_layer==4) & (y_origin==1))}")

                # 记录每一类样本的uncertainty
                self.LOGGER_MULTIVIEW_T.info("-----------clusters uncertainty-----------")
                mark_dict = {1:'normal_sample', 2:'easy_sample', 3:'hard_sample', 4:'outlier_sample'}
                for m in range(1, 5):
                    mean_u = np.mean(joint_opinion_cur_layer_origin[sample_marks_layer==m, -1])
                    std_u = np.std(joint_opinion_cur_layer_origin[sample_marks_layer==m, -1])
                    self.uncertainty_threshold[f"layer_{depth}_{mark_dict[m]}_mean"] = mean_u
                    self.uncertainty_threshold[f"layer_{depth}_{mark_dict[m]}_std"] = std_u
                    self.LOGGER_MULTIVIEW_T.info(f"  {mark_dict[m]}: mean uncertainty {mean_u:.4f}\t std uncertainty {std_u:.4f}")
                
            sample_marks_layer_df[f"layer_{depth}"] = sample_marks_layer

            # 层采样
            if self.cluster_samples_layer and sample_marks_layer is not None:
                if self.is_resample:
                    if self.layer_resample_method == "integration":
                        x_new_dict, y_new, (n_from_hard, n_from_outlier) = dl_train.resample_layer(
                            depth, 
                            sample_marks_layer, 
                            fold=(1+(1/(depth+1))),
                        )
                    else:
                        assert False, "parameter: \'layer_resample_method\' false"
                    # 如果采样的新样本数量不为0:
                    if len(y_new) != 0:

                        # 对每一层(注意要包括刚刚训练完成的这一层, depth这个标记是最外的for循环的最后才更新, 因此这里要+1)
                        for li in range(depth+1):

                            # 对每一层的每个view
                            for vi in range(self.n_view):
                                x_new_vi = x_new_dict[vi]
                                # 先生成增强特征
                                proba, opinion, _  = views[vi].layers[li].predict_opinion(x_new_vi)
                                proba = np.hstack(proba)
                                opinion = np.hstack(opinion)
                                if self.boost_feature_type == 'probability':
                                    boost_features = proba
                                elif self.boost_feature_type == False:
                                    boost_features = np.empty((len(x_new_vi), 0))

                                if self.is_stacking_for_boost_features or x_new_vi.shape[1]==dl_train.n_origin_features[vi]:
                                    # 如果設置stacking 或者 尚未添加過增強特徵, 则直接添加增强特征
                                    x_new_vi = np.hstack([x_new_vi, boost_features])
                                else:
                                    # 如果stacking=False, 则对增强特征做替换
                                    x_new_vi[:, dl_train.n_origin_features[vi]:dl_train.n_origin_features[vi]+boost_features.shape[1]] = boost_features

                                # 为第v个view的新样本构造完增强特征之后更新该view的新样本
                                x_new_dict[vi] = x_new_vi
                            
                        # 使用x_new_dict更新dl_train
                        for vi, x_new_vi in x_new_dict.items():
                            dl_train.update_samples_x_y(x_new_vi, y_new, vi, self.accumulation, n_train_origin_sample)

                    # 记录层采样的数量
                    self.LOGGER_MULTIVIEW_T.info(f"layer resample number: \t{len(y_new)}")
                    self.LOGGER_MULTIVIEW_T.info(f"number from hard: {n_from_hard}, number from outlier: {n_from_outlier} ")
                    print(f"  layer {depth} resample number: \t{len(y_new)}")
            
            # 计算本层的得分, 基于原始的样本计算
            opinion_fixed[~train_origin_mask] = joint_opinion_cur_layer_origin[~train_origin_mask]   # 固定opinion
            y_train_probas_avg = opinion_to_proba(joint_opinion_cur_layer_origin)

            current_evaluation = evaluate(y_origin, y_train_probas_avg)
            
            self.LOGGER_MULTIVIEW_T.info("  The evaluation[{}] of layer {} is {:.4f}".format(evaluate_name, depth, current_evaluation))
            self.LOGGER_MULTIVIEW_T.info("  The f1-score of layer {} is {:.4f}".format(depth, f1_macro(y_origin, y_train_probas_avg)))
            self.LOGGER_MULTIVIEW_T.info("  The accuracy of layer {} is {:.4f}".format(depth, accuracy(y_origin, y_train_probas_avg)))

            print("  The evaluation[{}] of layer {} is {:.4f}".format(evaluate_name, depth, current_evaluation))


            # 如果评估一层的综合性能时不考虑易分样本
            if (sample_marks_layer is not None) and evaluate_set == "trouble":
                # 如果 排除易分样本 进行评估
                mask_eval = sample_marks_layer != 2
                mask_eval = mask_eval[:n_origin_sample]
                cur_evaluation_not_easy = evaluate(y_origin[mask_eval], y_train_probas_avg[mask_eval])
                self.LOGGER_MULTIVIEW_T.info(\
                    "  The evaluation[{}](excluded easy samples) of layer {} is {:.4f}".format(evaluate_name, depth, cur_evaluation_not_easy))
                current_evaluation = cur_evaluation_not_easy
            elif (sample_marks_layer is None) and evaluate_set == "trouble":
                # 易份样本的划定要求生长的层数大于一定数量, 但最初的前span-1层没有办法划分易份样本, 
                # 因此要添加一个条件, 使得在基于非易份样本时, 不统计前span-1层的精度
                current_evaluation = 0


            # 评估是否需要更新最优层 
            if current_evaluation > best_layer_evaluation:
                best_layer_id = depth
                best_layer_evaluation = current_evaluation
            
            # 是否早停
            if depth-best_layer_id >= self.early_stop_rounds:
                break

            depth += 1

        # 训练结束后截断最佳模型, 从第0层到best_layer_id层
        self.best_layer_id = best_layer_id
        for view in views.values():
            view.keep_layers(0, best_layer_id+1)
        layers_opinions_list_origin = layers_opinions_list_origin[:best_layer_id+1]
        for v, op_list in views_opinions_origin_dict.items():
            views_opinions_origin_dict[v] = op_list[:best_layer_id+1]
            views_evidence_origin_dict[v] = views_evidence_origin_dict[v][:best_layer_id+1]
        sample_marks_layer_df = sample_marks_layer_df[[f"layer_{depth}" for depth in range(best_layer_id+1)]]
        self.LOGGER_MULTIVIEW_T.info("training finish")
        self.LOGGER_MULTIVIEW_T.info(\
            "best_layer: {}, current_layer:{}, save n_layers: {}\n".format(best_layer_id, depth, len(views[0])))
        print("best layer: {}, current layer:{}, save n layers: {}\n".format(best_layer_id, depth, len(views[0])))

        # 记录训练器的配置:
        self.LOGGER_MULTIVIEW_T.info(f"configs:")
        for v, view in enumerate(views.values()):
            self.LOGGER_MULTIVIEW_T.info(
                f"  view {v}:\t{[node.forest_type for node in view.layers[0].nodes]}")
        self.LOGGER_MULTIVIEW_T.info("Training ends\n\n\n")

        self.views = views
        self.layers_opinions_list_train = np.array(layers_opinions_list_origin)
        # 记录结果
        self.record_log("train", 
                        y_origin, 
                        views_opinions_origin_dict, 
                        np.array(layers_opinions_list_origin) ,
                        sample_marks_layer_df,
        )

        # 保存模型
        if self.is_save_model:
            self.save_model()

        # # 保存dl_train
        # import joblib
        # with open("/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/dl_train_beam_search_filter.pkl", "wb") as handle:
        #     joblib.dump(dl_train, handle)

    def predict(self, x: arr, y_test: arr):
        prob = self.predict_proba(x, y_test)
        label = self.category[np.argmax(prob, axis=1)]
        return label

    def predict_proba(self, x: arr, y_test: arr) -> arr:
        x_test_proba, _ = self.predict_opinion(x, y_test)
        return x_test_proba

    def predict_opinion(self, 
                        x: Dict[int, arr], 
                        y_test: arr,
                        bypass: bool=True,
                        use_views:str='all',
                        is_record:bool=True, 
                        record_suffix:str="", ) -> Tuple[arr, arr]:
        """

        Parameters
        ----------
        x: dict, shape=(#views, #samples, #features)
        """
        # 记录预测器的配置:
        self.LOGGER_MULTIVIEW_V.info(f"configs: ")
        for v, view in self.views.items():
            self.LOGGER_MULTIVIEW_V.info(
                f"  view {v}: {[node.forest_type for node in view.layers[0].nodes]}")

        # 评估预测效果
        evaluate = self.train_evaluation
        evaluate_name = evaluate.__name__

        x_test_dict = deepcopy(x)
        n_origin_feature_list = [item.shape[1] for item in x_test_dict.values()]        # 每个view的原始特征维度
        n_layer = len(self.views[0])
        n_origin_sample = x[0].shape[0]
        fixed_mask = np.zeros(n_origin_sample, dtype=bool)                              # 固定预测结果的掩码矩阵
        predict_proba = np.empty((n_origin_sample, self.n_class))
        predict_opinion = np.zeros((n_origin_sample, self.n_class+1)) + np.inf
        predict_evidence = np.empty((n_origin_sample, self.n_class))
        
        # 记录中间结果的变量
        # views_opinions_dict = {} #shape of (#layers, #views, #samples, #classes+1)
        views_opinions_dict = {f"view_{v}":[] for v in range(self.n_view)} #shape of (#views, #layers, #samples, #classes+1)
        layers_opinions_list = []   # shape of (#layers, #samples, #classes+1)
        sample_marks_layer_df = pd.DataFrame()  # 有可能第0层是没有做样本划分, 因此需要用DataFrame, 可以用索引

        for depth in range(n_layer):
            self.OPINION_LOGGER_V.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(depth))
            self.LOGGER_MULTIVIEW_V.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(depth))        
            
            view_opinion_list = []  # 第index层所有view的opinion
            
            intra_dict = {}
            # 预测
            # 1_n-1 层
            if depth < n_layer-1:
                prediction_confidence = 0
                # 对一层中的所有view
                for v in range(self.n_view):

                    # 预测
                    proba_list, opinion_list, evidence_list = self.views[v].predict_use_layer(
                        x_test_dict[v], depth, self.view_opinion_generation_method)
                    
                    # 更新测试数据集的特征
                    # 增强特征
                    x_test_dict[v] = self.update_features(                              
                        x_test_dict[v], y_test, proba_list, opinion_list, n_origin_feature_list[v], depth, v)

                    # 计算第i个view的联合opinion
                    opinion_view_i = self.generate_view_opinion(
                        opinion_list,
                        self.view_opinion_generation_method,
                    )   # shape = (#samples, #class+1)
                    # opinion_view_i = np.mean
                    # 计算第i个view的预测置信度
                    proba_view_i = opinion_view_i[:, :-1]
                    prediction_confidence += self.bypass_weights[depth][v] *proba_view_i
                    opinion_view_i[:, -1] = np.max(prediction_confidence, axis=1)
                    view_opinion_list.append(opinion_view_i)

                    # 记录LOG
                    self.LOGGER_MULTIVIEW_V.info("  The evaluation[{}] of view_{} is {:.4f}".format(
                        evaluate_name, v, evaluate(y_test, opinion_to_proba(opinion_view_i))))
            # 中间层
            else:
                # 对最后一层中的所有view,
                for v in range(self.n_view):
                    if (np.any(np.isnan(x_test_dict[v]))):
                        print(v)
                    # 预测
                    proba_list, opinion_view_i, _ = self.views[v].predict_use_layer(
                        x_test_dict[v], depth, self.view_opinion_generation_method)
                    
                    
                    view_opinion_list.append(opinion_view_i)

                    self.LOGGER_MULTIVIEW_V.info("The evaluation[{}] of view_{} is {:.4f}".format(
                            evaluate_name, v, evaluate(y_test, opinion_to_proba(opinion_view_i))))
                    self.LOGGER_MULTIVIEW_V.info("  The f1-score of view_{} is {:.4f}".format(
                            v, f1_macro(y_test, opinion_to_proba(opinion_view_i))))
                    self.LOGGER_MULTIVIEW_V.info("  The accuracy of view_{} is {:.4f}".format(
                            v, accuracy(y_test, opinion_to_proba(opinion_view_i))))
            
            # 获得本层基于所有view的联合opinion
            opinion_layer = joint_multi_opinion(view_opinion_list)
            # enable layerwise_opinion 
            if self.use_layerwise_opinion and depth>0:
                opinion_layer = joint_multi_opinion(
                    [ layers_opinions_list[-1], opinion_layer ]
                )
            
            # 本层的预测概率
            proba_layer = opinion_layer[:, :-1]

            # 更新预测结果
            # 如果对象中注册了bypass的属性, 则使用bypass属性
            if hasattr(self, "bypass_prediction"):
                bypass = self.bypass_prediction
            

            activate_mask = ~fixed_mask # 尚未固定的样本

            # 将尚未固定的样本的预测结果用新的来覆盖
            predict_opinion[activate_mask] = opinion_layer[activate_mask]
            predict_proba[activate_mask] = proba_layer[activate_mask]
            
            # bypass, 更新本层固定的高置信度的预测
            pre_fixed_mask = fixed_mask
            if bypass:
                if (depth < n_layer-1):
                    # 本层需要固定的样本
                    prediction_confidence = np.max(prediction_confidence, axis=1)
                    u_lower_mask = np.zeros(n_origin_sample, dtype=bool)
                    u_lower_mask[prediction_confidence > self.bypass_threshold*self.n_view] = True
                    # 更新固定好的样本
                    fixed_mask = fixed_mask | u_lower_mask
            cur_fixed_mask = fixed_mask & (~pre_fixed_mask)
            
            layers_opinions_list.append(predict_opinion.copy())

            sample_marks_layer_df[f"layer_{depth}"] = np.ones(n_origin_sample)

            ##### 日志记录           
            # 存储一组view的opinion
            for v, op in enumerate(view_opinion_list):
                views_opinions_dict[f"view_{v}"].append(op.copy())
            # 固定样本的数量
            self.LOGGER_MULTIVIEW_V.info(
                f"new fixed-sample number:\t{np.sum(cur_fixed_mask)}"
            )
            self.LOGGER_MULTIVIEW_V.info(
                f"total fixed-sample number:\t{np.sum(fixed_mask)} "
            )

            if bypass:
                # (总的)被判定为易分样本的识别准确度
                if np.sum(fixed_mask) != 0:
                    self.LOGGER_MULTIVIEW_V.info(f"[Total] The evaluation[{evaluate_name}] of easy sample:\t{evaluate(y_test[fixed_mask], predict_proba[fixed_mask]):.4f}" )
                
                # (本层的)被判定为易分样本的识别准确度
                if np.sum(cur_fixed_mask) != 0:
                    self.LOGGER_MULTIVIEW_V.info(f"[Current Layer] The evaluation[{evaluate_name}] of easy sample:\t{evaluate(y_test[cur_fixed_mask], predict_proba[cur_fixed_mask]):.4f}" )

            # 所有样本的识别准确度
            self.LOGGER_MULTIVIEW_V.info("The evaluation[{}] of layer {} with fixed is {:.4f}".format(
                evaluate_name, depth, evaluate(y_test, predict_proba)))
            self.LOGGER_MULTIVIEW_V.info("The evaluation[{}] of layer {} without fixed is {:.4f}".format(
                evaluate_name, depth, evaluate(y_test, proba_layer)))
            with open(os.path.join(self.logger_path, 'performance_per_layer_predict.txt'), 'a+') as file:
                file.write(
                    f'{evaluate_name}_layer{depth}: ' +
                    str(
                        evaluate(y_test, proba_layer)
                    ) + '\n'
                )
                       
            # 保存本层综合的opinion
            for v in range(self.n_view):
                self.OPINION_LOGGER_V.info(f"view opinion:\tview: {v}, \n{view_opinion_list[v]}")
            self.OPINION_LOGGER_V.info(
                f"layer opinion origin:\tlayer: {depth}, \n{opinion_layer}")
            self.OPINION_LOGGER_V.info(
                f"layer opinion with fixed:\tlayer: {depth}, \n{predict_opinion}")

        # 预测完成后, 将中间结果保存到成员变量中, 以便于后续其他的处理
        self.layers_opinions_list_predict = np.array(layers_opinions_list)
        self.layer_view_opinion_dict_predict = views_opinions_dict  # shape of (#views, )

        # record logs
        if is_record and y_test is not None:
            self.record_log("predict", 
                            y_test, 
                            views_opinions_dict, 
                            np.array(layers_opinions_list),
                            sample_marks_layer_df, 
                            record_suffix)

        return predict_proba, predict_opinion

    def update_features(self, x: arr, y: arr, proba: List[arr], opinion: List[arr], n_feature,
                        layer_id: int, view_id: int, ):
        """更新x, 将增强特征加到原始特征上
        Parameters
        ----------
        x: np.ndarray, shape=(#samples, #features)
        proba: ndarray, shape=(#samples, #classes * #nodes) or 
            (#nodes, #samples, #classes)
        opinion: ndarray, shape=(#samples, #classes * #nodes) or 
            (#nodes, #samples, #classes)
        n_feature: int, 原始特征数量

        Returns
        -------
        x: ndarray
        """
        stage = "predict"
        if len(np.shape(proba)) == 3:
            proba = np.hstack(proba)
        if len(np.shape(opinion)) == 3:
            opinion = np.hstack(opinion)
        n_node = len(self.estimator_configs)
        n_class = int(proba.shape[1] / n_node)

        if self.boost_feature_type == 'probability':
            boost_features = proba
        else: 
            boost_features = np.empty((x.shape[0], 0))
        # 记录相关性
        self.record_boost_features_corr(layer_id, view_id, boost_features, y, stage)
        if self.is_stacking_for_boost_features or (x.shape[1]==n_feature):
            x = np.hstack((x, boost_features))
        else:
            x[:, n_feature:n_feature+boost_features.shape[1]] = boost_features
        return x

    def metric(self, y_true, y_proba, evidence):
        """计算基于多种指标上计算得分"""
        f1:float = f1_macro(y_true, y_proba)
        acc:float = accuracy(y_true, y_proba)
        mse_l:float = mse_loss(y_true, evidence, 5, 10, True, self.n_class)
        return f1, acc, mse_l

    def preprocess(self, x_train, y_train, view=0) -> Tuple[arr, int, int]:
        x_train = x_train.reshape((x_train.shape[0], -1))
        category = np.unique(y_train)
        self.category = category
        # print(len(self.category))
        n_feature = x_train.shape[1]
        n_label = len(np.unique(y_train))
        self.LOGGER_MULTIVIEW_T.info(
            "####################  Begin to train  #######################")
        self.LOGGER_MULTIVIEW_T.info(
            "the shape of training samples: {}".format(x_train.shape))
        self.LOGGER_MULTIVIEW_T.info("use {} as training evaluation".format(
            self.train_evaluation.__name__))
        self.LOGGER_MULTIVIEW_T.info("stacking: {}, save model: {}".format(
            self.is_stacking_for_boost_features, self.is_save_model))
        bft = "opinions" if self.boost_feature_type else "proba"
        self.LOGGER_MULTIVIEW_T.info(f"boost feautres type: {bft}")
        return x_train, n_feature, n_label

    def group_samples_base_span(self, depth: int, opinion_list: List[arr], y_true: arr=None) -> arr:
        """划分样本
        对训练集: 同时考虑uncertainty 和 预测结果

        Parameters
        ----------
        y_true: ndarray of shape (#samples, ), 样本的真实标签
        depth: int,
        opinion_list: List[ndarray], shape is (span, #samples, #classes + 1)

        Returns
        -------
        marks: ndarray of shape (#samples, ). 样本的标记, 包含了1, 2, 3, 4.
            每一个样本可能的标记有 "normal"-1, "easy"-2, "hard"-3, "outlier"-4
        -------
        """
        if y_true is not None:
            span = self.span
            n_class = np.shape(opinion_list[0])[1]-1
            pred_list = \
                np.transpose([np.argmax(opinion[:, :-1], axis=1) for opinion in opinion_list])
            y_pred = np.argmax(np.sum(opinion_list, axis=0)[:, :-1], axis=1)

            uncertainty_list = np.atleast_2d(
                np.transpose([opinion[:, -1] for opinion in opinion_list]).squeeze()).reshape((len(y_pred),-1))
            uncertainty_list = \
                np.where(uncertainty_list > np.mean(uncertainty_list, axis=0), 1, 0)     # 低不确定度为0, 高不确定度为1
            unertainty_changed = \
                np.sum(uncertainty_list, axis=1) != np.max(uncertainty_list, axis=1) * span         # uncertainty改变的mask                                    
            unertainty_unchanged = \
                np.sum(uncertainty_list, axis=1) == np.max(uncertainty_list, axis=1) * span         # uncertainty未改变的mask
                
            low_uncertainty = unertainty_unchanged & (np.max(uncertainty_list, axis=1) == 0)    # 低不确定度的mask
            high_uncertainty = unertainty_unchanged & (np.max(uncertainty_list, axis=1) == 1)    # 高不确定度的mask

            pred_changed = np.sum(pred_list, axis=1) != np.max(pred_list, axis=1) * n_class # pred改变的mask
            pred_unchanged = pred_changed == False  # pred未改变的为mask

            pred_correct = (y_pred == y_true) & pred_unchanged     # 预测正确是为True
            pred_error = (y_pred != y_true) & pred_unchanged       # 预测错误为True

            marks = np.ones_like(y_pred)
            marks[low_uncertainty & pred_correct] = 2
            marks[high_uncertainty & pred_error] = 3
            marks[low_uncertainty & pred_error] = 4
        else:
            marks = self.group_samples_for_test(depth, opinion_list)
        return marks
    
    def group_samples_for_test(self, depth: int, opinion_list: List[arr]):
        """在测试阶段划分样本类型
        
            对预测集(测试集): 仅考虑uncertainty的趋势
                如果连续span层低于平均水平, 则是easy
                如果连续span层高于平均水平, 则是hard
                如果连续span层内发生波动, 则是normal
        """
        span = self.span
        if depth<span-1:
            raise ValueError("depth mask be greater than self.span-1")
        
        if len(opinion_list)>span:
            opinion_list=opinion_list[-span:]
        n_sample = len(opinion_list[0])
        uncertainty = np.array(opinion_list)[:, :, -1]  # shape of (#span, #sample)
        if depth <= span-1:
            easy_mean = [self.uncertainty_threshold[f"layer_{depth}_easy_sample_mean"] for _ in range(span)]
            hard_mean = [self.uncertainty_threshold[f"layer_{depth}_hard_sample_mean"] for _ in range(span)]
        else:
            easy_mean = [self.uncertainty_threshold[f"layer_{di}_easy_sample_mean"] for di in range(depth-span+1, depth)] 
            hard_mean = [self.uncertainty_threshold[f"layer_{di}_hard_sample_mean"] for di in range(depth-span+1, depth)]

        easy_mean = np.reshape(easy_mean, (-1, 1))

        hard_mean = np.reshape(hard_mean, (-1,1))
        marks = np.ones(n_sample)
        # easy_sample
        marks[np.all(uncertainty<easy_mean, axis=0)] = 2
        # hard smaple
        marks[np.all(uncertainty>hard_mean, axis=0)] = 3
        return marks

    def group_samples_by_uncertainty(self, layer_id: int, opinion: arr, y_true: arr=None) -> arr:
        """根据不确定度划定样本类型
            将uncertainty转化为confidence, 
            对confidence作排序, 计算截断的错误率
            """
        def find_first_error_below_threshold(binary_array, threshold):
            binary_array = binary_array.astype(bool)
            total_count = np.arange(1, len(binary_array) + 1)
            error_count = np.cumsum(1 - binary_array)
            error_rate_array = error_count / total_count
            indices = np.where(error_rate_array < threshold)[0]
            if indices.size > 0:
                last_index_below_threshold = np.max(indices)        # 索引最大的, 就是confidence最小的
                return last_index_below_threshold
            else:
                return None

        n_class = np.shape(opinion)[1]-1
        y_pred = np.argmax(opinion[:, :n_class], axis=1)
        uncertainty_array = opinion[:, -1]

        marks = np.ones_like(y_pred)

        if y_true is not None:
            
            confidence_array = 1-uncertainty_array
            binary_array = np.array(y_true==y_pred, dtype=bool) # 真实样本与预测样本一致则为True, 

            err_mean = np.sum(~binary_array)/binary_array.size  # 训练集的错误率

            ids_sorted = np.argsort(confidence_array)[::-1]         # 对confidence降序排序的索引
            threshold_idx = find_first_error_below_threshold(binary_array, self.alpha*err_mean)       # 找到 最佳不确定度阈值对应的索引
            if threshold_idx is None:
                u_threshold = 0
            else:
                u_threshold = uncertainty_array[ids_sorted[threshold_idx]]# 不确定度的阈值
            self.uncertainty_threshold[f"layer_{layer_id}"] = u_threshold

            # 大于threshold, 为high_uncertainty, 小于threshold, 为low_uncertainty
            # 必须要加 squeeze，不然有可能报bug
            low_uncertainty = np.squeeze(uncertainty_array <= u_threshold)
            high_uncertainty = np.squeeze(uncertainty_array > u_threshold)
        
            pred_correct = binary_array     # 预测正确是为True
            pred_error = ~binary_array      # 预测错误为True
            # print(f"correct prediction number: {np.sum(pred_correct)}")
            try:
                marks[low_uncertainty & pred_correct] = 2
                marks[high_uncertainty & pred_error] = 3
                marks[low_uncertainty & pred_error] = 4
            except IndexError:
                for i,item in enumerate([low_uncertainty, high_uncertainty, pred_correct]):
                    print(f"{i}: {item}")
                raise IndexError
        else:
            low_uncertainty = uncertainty_array <= self.uncertainty_threshold[f"layer_{layer_id}"]
            high_uncertainty = uncertainty_array > self.uncertainty_threshold[f"layer_{layer_id}"]
            marks[low_uncertainty] = 2  # 易份样本
            marks[high_uncertainty] = 3 # 高不确定度为难分样本

        return marks

    def group_samples_by_uncertainty_bins(self, layer_id: int, opinion: arr, y_true: arr=None) -> arr: 
        """ base DCB-Forest """

        def get_acc(b_array:np.array):
            """b_array: 二进制数组, 正确预测为1, 错误预测为0"""
            return np.sum(b_array)/b_array.size
        def split_array_into_bins(arr, k):
            """将给定的数组等分为 k 个箱子，并返回每个箱子中的元素"""
            n = len(arr)
            bin_size = n // k
            remainder = n % k

            bins = []
            start = 0
            for i in range(k):
                if i < remainder:
                    end = start + bin_size + 1
                else:
                    end = start + bin_size
                bins.append(arr[start:end])
                start = end

            return bins
        
        n_bins = self.n_bins    # default 30
        n_class = np.shape(opinion)[1]-1
        y_pred = np.argmax(opinion[:, :n_class], axis=1)
        uncertainty_array = opinion[:, -1]

        marks = np.ones_like(y_pred)
        if y_true is not None:
            ta = self.ta
            binary_array = np.array(y_true==y_pred, dtype=bool) # 真实样本与预测样本一致则为True,
            confidence_array = 1-uncertainty_array
            ids_sorted = np.argsort(confidence_array)[::-1]         # 对confidence降序排序的索引
            bins_ids = split_array_into_bins(ids_sorted, n_bins)
            for k, bi in enumerate(bins_ids):
                acc = get_acc(binary_array[bi])
                if acc<ta:
                    if k==0:
                        threshold_idx = None
                    else:
                        threshold_idx = bins_ids[k-1][-1]
                    break
            try:
                if threshold_idx is None:
                    u_threshold = 0
                else:
                    u_threshold = uncertainty_array[threshold_idx]
            except UnboundLocalError:
                u_threshold = 0
            self.uncertainty_threshold[f"layer_{layer_id}"] = u_threshold

            low_uncertainty = np.squeeze(uncertainty_array <= u_threshold)
            high_uncertainty = np.squeeze(uncertainty_array > u_threshold)
            pred_correct = binary_array     # 预测正确是为True
            pred_error = ~binary_array      # 预测错误为True

            try:
                marks[low_uncertainty & pred_correct] = 2
                marks[high_uncertainty & pred_error] = 3
                marks[low_uncertainty & pred_error] = 4
            except IndexError:
                for i,item in enumerate([low_uncertainty, high_uncertainty, pred_correct]):
                    print(f"{i}: {item}")
                raise IndexError
        else:
            low_uncertainty = uncertainty_array <= self.uncertainty_threshold[f"layer_{layer_id}"]
            high_uncertainty = uncertainty_array > self.uncertainty_threshold[f"layer_{layer_id}"]
            marks[low_uncertainty] = 2  # 易份样本
            marks[high_uncertainty] = 3 # 高不确定度为难分样本

        return marks
    
    def eval_new_featurs(self, x_new, y):
        """评估新特征"""
        # x_new = 
        pass

    def record_log(self, stage, y_true, 
                         views_opinions_dict,
                         layers_opinions_list,
                         sample_marks_layer_df,
                         record_suffix:str="", ):
        """记录训练的一些中间结果"""
        # n_l = len(layers_opinions_list_train)
        # n_v = len(views_opinions_origin_dict)
        

        ###### 训练阶段记录树与森林的相关信息 ##### 
        if stage == "train":
            ######## 叶子平均数量
            mean_leaves = \
            np.mean( [ [ [ [ [ dt.tree_.n_leaves for dt in estimator.estimators_] for estimator in node.estimators_.values()] for node in layer.nodes] for layer in view.layers] for view in self.views.values() ] , axis = (3,4))  # shape of ()
            self.__n_v, self.__n_l, self.__n_no = mean_leaves.shape
            mean_leaves = np.concatenate(mean_leaves, axis=1)
            layerIndex = [f"layer_{l}" for l in range(self.__n_l)]
            viewIndex = [f"view_{v}" for v in range(self.__n_v)]
            nodeColumns = [f"view_{v}_node_{n}" for v in range(self.__n_v) for n in range(self.__n_no)]
            nodeIndex = pd.MultiIndex.from_product([["n leaves"], layerIndex])    # 二级索引, [n leaves, layer_{l}]
            self.mean_leaves_df = pd.DataFrame(mean_leaves, index=nodeIndex, columns=nodeColumns)
            self.mean_leaves_df.to_csv(self.logger_path + "/tree_info.csv", mode='a')

            ######## 决策树平均深度
            mean_max_depth = \
            np.mean( [ [ [ [ [ dt.tree_.max_depth for dt in estimator.estimators_] for estimator in node.estimators_.values()] for node in layer.nodes] for layer in view.layers] for view in self.views.values() ] , axis = (3,4))
            mean_max_depth = np.concatenate(mean_max_depth, axis=1)
            depthIndex = pd.MultiIndex.from_product([["depth"], layerIndex])
            self.mean_max_depth_df = pd.DataFrame(mean_max_depth, index=depthIndex, columns=nodeColumns)    # 二级索引, [depth, layer_{l}]
            self.mean_max_depth_df.to_csv(self.logger_path + "/tree_info.csv", mode='a')
        else:
            layerIndex = [f"layer_{l}" for l in range(self.__n_l)]
            viewIndex = [f"view_{v}" for v in range(self.__n_v)]

        if stage == "train" :
            file_suffix = "train" 
        elif stage == "predict":
            file_suffix = "predict"
        
        if file_suffix != "":
            file_suffix += "_"+record_suffix

        ######## 每层每个view的精度  shape of (#views, #layers, #metrics)
        views_scores_train = \
        np.array(
            [ [ get_scores( y_true, opinion_to_proba(op) ) 
                for op in op_list ] for op_list in views_opinions_dict.values()]
        )

        ######## 每层联合opinion的精度 # shape of (#layers, #metrics)
        joint_layers_scores_train = \
        np.array( 
            [ get_scores( y_true, opinion_to_proba(op) ) 
              for op in layers_opinions_list ]
        )
        ####### 每层平均opinion的精度    
        mean_opinion_list = np.array( [ np.mean(op_tup, axis=0) #shape of (#layers, )
                        for op_tup in zip( *(op_list for op_list in views_opinions_dict.values()) ) ]
        ) # shape of (#layers, #metrics)
        ave_layers_scores_train = np.array( 
            [ get_scores( y_true, opinion_to_proba(op) )
              for op in mean_opinion_list ]
        )

        for n_me, name in enumerate(metrics_name) :
            multiIndex = pd.MultiIndex.from_product([[name], layerIndex])
            scores = views_scores_train[:, :, n_me].T   # 转置之后的shape (#layers, #views)
            score_df = pd.DataFrame(scores, 
                                    index=multiIndex, 
                                    columns=viewIndex
            )
            score_df["joint"] = joint_layers_scores_train[:, n_me]
            score_df["mean"] = ave_layers_scores_train[:, n_me]
            score_df.to_csv(self.logger_path+f"/{file_suffix}_info.csv", mode="a")

        ####### bypass的view精度 ######
        def replace(layer_id, old_arr, new_arr):
            if layer_id == 0:
                return old_arr
            ids = sample_marks_layer_df[f"layer_{layer_id-1}"].values==2    # 易分样本索引
            replaced_arr = old_arr.copy()         
            replaced_arr[ids] = new_arr[ids]    # 保留易分样本结果
            return replaced_arr
        
        views_scores_train_bypass_layer = np.array(
            [ [ get_scores( 
                    y_true,  
                    replace(
                        i,
                        opinion_to_proba(op),
                        opinion_to_proba(layers_opinions_list[i-1]),
                    )
                ) 
                for i, op in enumerate(op_list)
                ]
            for op_list in views_opinions_dict.values()
            ]
        )

        ####### bypass平均opinion的精度    
        mean_opinion_list = np.array( [ np.mean(op_tup, axis=0) #shape of (#layers, )
                        for op_tup in zip( *(op_list for op_list in views_opinions_dict.values()) ) ]
        ) # shape of (#layers, #metrics)
        ave_layers_scores_train_bypass = np.array( 
            [ get_scores( 
                y_true, 
                replace(
                    i, 
                    opinion_to_proba(op), 
                    opinion_to_proba(mean_opinion_list[i-1]),
                ),
              )
            for i, op in enumerate(mean_opinion_list) ]
        )

        views_scores_train_bypass_view = np.array(
            [ [ get_scores( 
                    y_true,  
                    replace(
                        i,
                        opinion_to_proba(op),
                        opinion_to_proba(op_list[i-1]),
                    )
                ) 
                for i, op in enumerate(op_list)
                ]
            for op_list in views_opinions_dict.values()
            ]
        )


        ######## trouble sample 的 view 的精度 ######
        views_scores_train_trouble = np.array(
            [ [ get_scores( 
                    y_true[ sample_marks_layer_df[f"layer_{i}"].values!=2 ],  
                    opinion_to_proba(op)[sample_marks_layer_df[f"layer_{i}"].values!=2] ) 
                for i, op in enumerate(op_list)
                ]
                for op_list in views_opinions_dict.values()
            ]
        )

        ######## 每层trouble sample联合opinion的精度 # shape of (#layers, #metrics)
        joint_layers_scores_train_trouble = \
        np.array( 
            [ get_scores( y_true[sample_marks_layer_df[f"layer_{i}"].values!=2], 
                          opinion_to_proba(op)[sample_marks_layer_df[f"layer_{i}"].values!=2] ) 
              for i, op in enumerate(layers_opinions_list) 
            ]
        )

        ####### 每层trouble sample平均opinion的精度    
        mean_opinion_list = np.array( [ np.mean(op_tup, axis=0) #shape of (#layers, )
                        for op_tup in zip( *(op_list for op_list in views_opinions_dict.values()) ) ]
        ) # shape of (#layers, #metrics)
        ave_layers_scores_train_trouble = np.array( 
            [ get_scores( y_true[sample_marks_layer_df[f"layer_{i}"].values!=2], opinion_to_proba(op)[sample_marks_layer_df[f"layer_{i}"].values!=2] )
              for i, op in enumerate(mean_opinion_list) ]
        )

        ####### 记录bypass的精度到csv文件-用每个view的结果bypass
        for n_me, name in enumerate(metrics_name) :
            multiIndex = pd.MultiIndex.from_product([["bypass_view_"+name], layerIndex])
            scores = views_scores_train_bypass_view[:, :, n_me].T   # 转置之后的shape (#layers, #views)
            score_df = pd.DataFrame(
                scores, index=multiIndex, columns=viewIndex
            )
            score_df.to_csv(self.logger_path+f"/{file_suffix}_info.csv", mode="a")


        ####### 记录bypass的精度到csv文件-用joint的结果bypass
        for n_me, name in enumerate(metrics_name) :
            multiIndex = pd.MultiIndex.from_product([["bypass_layer_"+name], layerIndex])
            scores = views_scores_train_bypass_layer[:, :, n_me].T   # 转置之后的shape (#layers, #views)
            score_df = pd.DataFrame(
                scores, index=multiIndex, columns=viewIndex
            )
            score_df["joint"] = joint_layers_scores_train[:, n_me]
            score_df["mean"] = ave_layers_scores_train_bypass[:, n_me]
            score_df.to_csv(self.logger_path+f"/{file_suffix}_info.csv", mode="a")


        ####### 记录trouble sample的精度 到csv文件
        for n_me, name in enumerate(metrics_name) :
            multiIndex = pd.MultiIndex.from_product([["trouble_"+name], layerIndex])
            scores = views_scores_train_trouble[:, :, n_me].T   # 转置之后的shape (#layers, #views)
            score_df = pd.DataFrame(
                scores, index=multiIndex, columns=viewIndex
            )
            score_df["joint"] = joint_layers_scores_train_trouble[:, n_me]
            score_df["mean"] = ave_layers_scores_train_trouble[:, n_me]
            score_df.to_csv(self.logger_path+f"/{file_suffix}_info.csv", mode="a")


        ######## 记录每一层的opnion与标签值
        excel_handle = pd.ExcelWriter(
            self.logger_path + f"/prediction_all_layer_{file_suffix}.xlsx",
            engine='openpyxl',)
        for layer_id, op in enumerate(layers_opinions_list):
            record = record_opinion_res(y_true, op, sample_marks_layer_df[f"layer_{layer_id}"])
            # record['marks'] = sample_marks_layer_df[f"layer_{layer_id}"]
            record.to_excel(excel_handle, sheet_name=f"layer_{layer_id}")
        excel_handle.close()
        ######## 记录最后一层的opinion和标签值
        record = record_opinion_res(y_true, layers_opinions_list[-1], sample_marks_layer_df[f"layer_{self.best_layer_id}"])
        record.to_csv(
            self.logger_path + f"/prediction_final_layer_{file_suffix}.csv"
        )

        ######## 记录每一层的每个模态的opinions与标签值
        excel_file = self.logger_path + f"/prediction_all_views_{file_suffix}.xlsx"
        if os.path.exists(excel_file):
            os.remove(excel_file)
        excel_handle = pd.ExcelWriter(
            self.logger_path + f"/prediction_all_views_{file_suffix}.xlsx",
            engine='openpyxl',)
        for l_id, layer_name in enumerate(layerIndex):
            layer_opinions = [op_list[l_id] for op_list in views_opinions_dict.values()]    # shape of (#n_views, #samples, #classes+1)
            col_names = pd.MultiIndex.from_product([viewIndex, ['belief_0', 'belief_1', 'uncertainty']])
            record = pd.DataFrame(np.concatenate(layer_opinions, axis=1), columns=col_names)
            record.to_excel(excel_handle, sheet_name=layer_name)
        excel_handle.close()

        # 存储每一层的样本类型标记
        sample_marks_layer_df.to_csv(
            self.logger_path + f"/sample_marks_{file_suffix}.csv"
        ) 
        # 存储 每一层每个样本的uncertainty
        u_train = layers_opinions_list[:, :, -1].T # 转置之后, shape (#samples, #layers)
        pd.DataFrame(
            u_train, 
            columns=layerIndex
        ).to_csv(
            self.logger_path + f"/uncertainty_all_layers_{file_suffix}.csv"
        )

        # 查看 是否有错误预测的样本在后续层的训练中被纠正
        pred_statement_train = get_stage_matrix(layers_opinions_list, y_true)
        pd.DataFrame(
            pred_statement_train, 
            columns=layerIndex
        ).to_csv(
            self.logger_path + f"/y_pred_statement_{file_suffix}.csv"
        )

    def __print_features_info(self, v:int, dl_train:DataLoader,):
        """打印 特征信息 """
        names = dl_train.boost_features_names[v]
        n_origin_ = dl_train.n_origin_features[v]
        n_boost_ = len([elem for elem in names if "boost" in elem])
        n_intra_ = len([elem for elem in names if "intra" in elem])
        n_inter_ = len([elem for elem in names if "inter" in elem])

        return n_origin_, n_boost_, n_intra_, n_inter_
    
    def generate_view_opinion(self, opinion_list, view_opinion_generation_method):
        """生成 view的综合opinion 
        Parameters
        ----------
        opinion_list: shape = (#nodes, #samples, #class+1)"""
        if view_opinion_generation_method == "mean":  # 平均node产出的opinion
            opinion_view = np.mean(opinion_list, axis=0)
            # opinion_view = np.mean(opinion_view, axis=)
        else:
            raise Exception("please set true parameter for \'view_opinion_generation_method\'")
        return opinion_view

    def cluster_sample(self, cluster_method:str, depth:int, opinions_list:List, y=None):
        """划分样本, 划分为4种类型的样本
        Parameters:
        -----------

        """
        if cluster_method == 'span':
            if depth+1>=self.span:
                sample_marks_layer = self.group_samples_base_span(depth, opinions_list[-self.span:], y)
            else:
                n_samples = opinions_list[0].shape[0]
                sample_marks_layer = np.ones(n_samples, dtype=int)
        else:
            raise ValueError("parameter: 'cluster_method setting error")
        return sample_marks_layer

    def __record_features_corr(self, features, y, filename, col_name):
        """记录高阶特征之间的相关性"""
        pass
        # if features.shape[1] == 0:
        #     return
        # import sys
        # sys.path.append("/home/tq/uncertainty_estimation_0403/MVUGCForest/FeatureImportance")
        # from Correlation import generate_corr_pic
        # dir = os.path.join(self.logger_path, "high_order_correlation")
        # if not os.path.exists(dir):
        #     os.mkdir(dir)
        
        # filename = os.path.join(dir, filename)

        # generate_corr_pic( np.hstack( [ features, y.reshape(-1,1) ] ),
        #                              filename, col_name=col_name)
        
    def record_inter_features_corr(self, depth, features, y, stage:str ):
        filename = f"correlation_inter_label_{stage}_layer_{depth}.png"
        col_name = [ f"inter_{i}_layer_{depth}" for i in range(features.shape[1]) ]
        col_name.append("label")
        self.__record_features_corr(features, y, filename, col_name)

    def record_intra_features_corr(self, depth, view, features, y, stage:str ):
        filename = f"correlation_intra_label_{stage}_layer_{depth}_view_{view}.png"
        col_name = [ f"intra_{i}" for i in range(features.shape[1]) ]
        col_name.append("label")
        self.__record_features_corr(features, y, filename, col_name)

    def record_boost_features_corr(self, depth, view, features, y, stage:str ):
        filename = f"correlation_boost_label_{stage}_layer_{depth}.png"
        col_name = [ f"boost_{i}" for i in range(features.shape[1]) ]
        col_name.append("label")
        self.__record_features_corr(features, y, filename, col_name)

    def save_model(self):
        """保存模型"""
        import joblib
        model_dir = os.path.join(self.logger_path, "model")
        if not os.path.exist(model_dir):
            os.mkdir(model_dir)
        with open(os.path.join(self.logger_path, "MVUGCForest_model.pkl"), "wb" ) as handle:
            joblib.dump(self, handle)