'''
Description: MVUGCForest的sklearn接口
Author: tanqiong
Date: 2023-07-22 21:31:22
LastEditTime: 2023-07-25 16:05:19
LastEditors: tanqiong
'''
from .MVUGCForest import MVUGCForest, aupr, auroc, accuracy, f1_macro
class MVUGCForestClassifier():
    def __init__(self, logger_path) -> None:
        config = self.get_config()
        self.model:MVUGCForest = MVUGCForest(config)
        pass

    def get_config(self, ):
        config={}
        
        # boost_features
        config["is_stacking_for_boost_features"] = False
        config["boost_feature_type"] = 'opinion'            # 'opinion', 'probability', None
        
        # high order features
        config["is_RIT"] = True
        config["is_intersection_inner_view"] = True
        config["is_intersection_cross_views"] = True
        config["enforce_intra"] = True                             # 是否强迫生成intra-section feature
        config["intra_feature_least"] = 1                     # 强迫生成intra-section的数量
        config["inter_method"] = "operator"                 # 'stack', 'operator'
        config["filter_intra"] = False
        config["filter_inter"] = True
        config["ts_encoding"] = False                       # TS编码 True('uncertainty'), False,  'opinion', 'uncertainty', 'proba'
    
        # cluster samples
        config["span"]=2
        config["cluster_method"] = "uncertainty_bins"                     # 'uncertainty', 'span', 'uncertainty_bins'
        config["n_bins"] = 30
        config["ta"] = 0.7
        
        # resample
        config["is_resample"] = True              # True/Fasle, conflict with 'is_resample'
        config["onn"] = 3                       # outlier nearest neighbour
        config["layer_resample_method"] = "integration"    # "integration", "respective"
        config["accumulation"] = False

        # training 
        config["max_layers"]=10
        config["early_stop_rounds"]=2    
        config["is_weight_bootstrap"] = True
        config["train_evaluation"] = aupr    # accuracy, f1_macro, aupr, auroc, mse_loss
        config["view_opinion_generation_method"] = "joint"  # 'mean'/'joint'/'sum' (sum==mean)
        config["is_save_model"] = False
        # config["random_state"]=666     # 669172976

        # prediction
        config["is_des"] = False
        config["use_layerwise_opinion"] = True              # 是否考虑层间联合opinion

        # node configs
        config["uncertainty_basis"] = "evidence"    # "entropy" / "evidence"
        config["evidence_type"] = "probability"     # "knn"(舍弃) / "probability"
        config["act_func"] = "approx_step"                 # 'approx_step', 'ReLU', None
        config["W_type"] = "sum"               # 'n_class', 'n_tree', 'sum', 'variable'
        config["use_kde"] = False               # 是否使用kde以额外考虑数据不确定度
        config["estimator_configs"]=[]
        for _ in range(2):
            config["estimator_configs"].append({"n_fold":3, 
                                                "type": "RandomForestClassifier",
                                                ### sklearn parameters ###
                                                "n_estimators": 50, 
                                                "max_features": 'sqrt',   # None/'sqrt'/float
                                                # "max_samples": 0.7,
                                                "max_depth": None, 
                                                "n_jobs": -1, 
                                                "min_samples_leaf": 2,
                                                # "min_weight_fraction_leaf": 2/504,
                                                # "max_features": None,
                                                # "bootstrap": False,
                                                })
        for _ in range(2):
            config["estimator_configs"].append({"n_fold": 3, 
                                                "type": "ExtraTreesClassifier",
                                                ### sklearn parameters ###
                                                "n_estimators": 50, 
                                                "max_features": 'sqrt',  # None/'sqrt'/float
                                                # "max_samples": 0.7, 
                                                "max_depth": None, 
                                                "n_jobs": -1, 
                                                "min_samples_leaf": 2, 
                                                # "min_weight_fraction_leaf": 2/504,
                                                # "max_features": None,
                                                # "bootstrap": False,
                                                })
        return config

    def fit(self, x_train, y_train):
        """
        x_train shape: (#views, #samples, #features)
        """
        self.model.fit_multiviews(x_train, y_train)
        return self.model
    
    def predict(self, x_test, y_test=None):
        if (y_test is None):
            y_test = [0 for _ in x_test]
        y_pred = self.model.predict(x_test, y_test)
        return y_pred
    
    def predict_proba(self, x_test, y_test=None):
        if (y_test is None):
            y_test = [0 for _ in x_test]
        y_proba = self.model.predict_proba(x_test, y_test)
        return y_proba
    
    def predict_opinion(self, x_test, y_test=None):
        if (y_test is None):
            y_test = [0 for _ in x_test]
        proba_pred, opinion_pred = self.model.predict_opinion(x_test, y_test)
        return opinion_pred