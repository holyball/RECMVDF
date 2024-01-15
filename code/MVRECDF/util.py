import numpy as np
import pandas as pd
from scipy import stats
from numpy import ndarray as arr
from typing import List, Tuple, Union, Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaseEnsemble
from sklearn.metrics import euclidean_distances
from .evaluation import aupr, auroc, f1_macro, accuracy
from .uncertainty import opinion_to_proba, opinion_to_label

metrics_name = [func.__name__ for func in [auroc, aupr, f1_macro, accuracy]]

def record_opinion_sorted_by_uncertainty(y_true: arr, opinion: arr, ascending:bool = True):
    """按按照不确定度升序对样本排序, 记录:样本索引, 真实标签, 预测标签, 不确定度
    
    Parameters
    ----------
    y_true: 
    opinion: 
    ascending: 依据不确定度对样本作升序排序
    
    Returns
    -------
    record_df: DataFrame of shape (#samples, 3), sorted by uncertainty(ascending)
               columns: ['y_true', 'y_pred', 'is_correct','belief_0', 'belief_1', 'uncertainty']
    """
    assert y_true.shape[0] == opinion.shape[0], "样本数量不匹配"
    proba = opinion_to_proba(opinion)
    y_pred = np.argmax(proba, axis=1)
    uncertainty = opinion[:,-1]
    if ascending:
        idx_sorted = np.argsort(uncertainty)
    else:
        idx_sorted = np.arange(len(y_true))
    record = np.transpose([y_true[idx_sorted], 
                           y_pred[idx_sorted], 
                           y_true[idx_sorted]==y_pred[idx_sorted]])
    record = np.hstack([record, opinion[idx_sorted]])
    column_names = ['y_true', 'y_pred', 'is_correct','belief_0', 'belief_1', 'uncertainty']
    record_df = pd.DataFrame(record, index=idx_sorted, columns=column_names)
    return record_df

def record_opinion_res(y_true:arr, pred_opinion:arr, marks:arr):
    """记录预测结果是否正确"""
    record_df = record_opinion_sorted_by_uncertainty(y_true, pred_opinion, ascending=False)
    record_df["marks"] = marks[record_df.index.astype(int)]
    return record_df

def df_to_csv(df: pd.DataFrame, dir:str, filename:str):
    try:
        import os
        if not os.path.exists(dir):
            os.mkdir(dir)
        df.to_csv(os.path.join(dir, filename))
    except:
        raise FileNotFoundError(f"cannot find the dictionary {dir}")
    else:
        return True

def get_stage_matrix(opinion_list, y_true):
    """查看每层预测的情况
    Parameters
    ----------
    opinion_list: 
    y_true: 

    Returns
    -------
    pred_statement: ndarray, shape of (#samples, #layers)

    Examples:
    --------
    pred_statement = get_stage_matrix(gc.layers_opinions_list_train, y_train)
    
    pred_statement: shape of (#samples, #layers)
    """
    n_class = len(np.unique(y_true))
    y_pred_train_list = [np.argmax(opinion[:,:n_class], axis=1) for opinion in opinion_list]
    pred_statement = [y_true==y_pred for y_pred in y_pred_train_list]
    return np.transpose(pred_statement)

def get_scores(y_true, proba)->list:
    """计算四种分数"""
    score_list = [ auroc( y_true, proba ),
                   aupr( y_true, proba ),
                   f1_macro( y_true, proba ),
                   accuracy( y_true, proba ),
    ]
    return score_list

def record_trees_feature_indices(random_forest)->List:
    """记录随机森林中每一棵树使用的特征索引"""
    feature_indices = []
    for estimator in random_forest.estimators_:
        tree = estimator.tree_
        # idx = tree.feature
        feature_indices.append(np.unique(tree.feature)[1:])
    return feature_indices

def record_forest_train_samples(rf: Union[RandomForestClassifier, ExtraTreesClassifier], 
                                X_train: np.ndarray) -> Dict[int, List[int]]:
    tree_samples = {}
    for i, dt in enumerate(rf.estimators_):
        tree_samples[i] = record_dt_leaf_train_samples(dt, X_train)

    return tree_samples

def record_dt_leaf_train_samples(dt: DecisionTreeClassifier, X_train: np.ndarray):
    """记录决策树的每棵树的叶子的对应的训练集样本
    
    Returns
    -------
    leaf_samples: Dict(int, list), 叶节点的索引-叶节点对应的训练集特征

    Note
    ----
    由于MVUGCForest在训练时可能会存在人造样本, 而传入的X_train可能是原始样本, 
    所以决策树的叶子可能不会被X_train全部覆盖, 导致某些叶子结点的特征列表为空[]. 

    """
    leaf_indices = dt.apply(X_train)
    leaf_samples = {}
    # 初始化leaf_samples
    for node_idx, feature_idx in enumerate(dt.tree_.feature):
        if (feature_idx != -2):
            leaf_samples[node_idx] = []
    # 更新leaf_samples
    for i, leaf_index in enumerate(leaf_indices):
        if leaf_index not in leaf_samples:
            leaf_samples[leaf_index] = []
        leaf_samples[leaf_index].append(i)

    return leaf_samples

def record_forest_features(rf: Union[RandomForestClassifier, ExtraTreesClassifier]):
    tree_features = {}
    for i, dt in enumerate(rf.estimators_):
        tree_features[i] = record_dt_leaf_features(dt)
    
    return tree_features

def record_dt_leaf_features(dt: DecisionTreeClassifier):
    """获取 一颗决策树 所有叶子结点的路径
    
    Returns
    -------
    leaf_features: Dict(int, list), 叶节点的索引-叶节点对应路径的特征
    """
    leaf_features = {}

    def traverse_tree(node_id, feature_indices):
        if dt.tree_.children_left[node_id] == -1:  # 叶子节点
            leaf_id = node_id
            leaf_features[leaf_id] = list(feature_indices)
        else:
            feature_index = dt.tree_.feature[node_id]
            if feature_index != -2:  # 跳过非叶子节点
                feature_indices.add(feature_index)
            left_child_id = dt.tree_.children_left[node_id]
            traverse_tree(left_child_id, feature_indices.copy())
            right_child_id = dt.tree_.children_right[node_id]
            traverse_tree(right_child_id, feature_indices.copy())

    traverse_tree(0, set())
    
    return leaf_features

def get_des_weights(x_test: np.ndarray, 
                    y_preds: list,
                    # tree_leaf_samples_dict: Dict,
                    # tree_leaf_features_dict: Dict,
                    # tree_features_dict: Dict,
                    tree_features_list: List,
                    x_train: np.ndarray,
                    y_train: np.ndarray,
                    )->np.ndarray:
    """获取Dynamic Ensemble Selection的权重

    Parameters
    ----------
    x_test: unknown data
    y_preds: shape of (#samples, #trees), 每个森林对每个样本的预测结果
    tree_features_list: List, 每棵树使用的特征的索引
    x_train: 训练集特征
    y_train: 训练集标签
    
    Returns
    -------
    weights: ndarray, shape of (#samples, #trees)
    
    """
    
    weights = []
    # tree_features_list = list(tree_features_dict.values())

    # weights = np.apply_along_axis(
    #     get_weights,
    #     axis=1,
    #     arr = x_test,
    #     y_preds=y_preds[0],
    #     x_train = x_train,
    #     y_train = y_train,
    #     feature_idx_list = tree_features_list,
    # )
    for x, y_pred in zip( x_test, y_preds ):
        weights.append( get_weights(x, y_pred, x_train, y_train, tree_features_list) )
    
    return np.array(weights)
        
def get_weights(x, y_preds, x_train, y_train, feature_idx_list):
    """计算每棵树应该具有的权重
    Parameters
    ----------
    x: ndarray, 单个样本
    y_pred_list: ndarray, 决策树对x的分类结果, (#trees, )
    x_train: ndarray, 训练集样本
    y_train: ndarray, 训练集的标签
    feature_idx_list: list, shape of (#trees, ), 特征索引

    Returns
    -------
    weights: list, (#tree, )
    """
    x = np.atleast_2d(x)
    x_train = np.atleast_2d(x_train)

    weights = []
    for tree_id, feature_ids in enumerate(feature_idx_list):
        x_refer = x_train[ y_train == y_preds[tree_id], : ][ :, feature_ids ]
        if ( len(x_refer) == 0 or len(feature_ids)==0 ):
            weights.append(1/len(x_train))
        else:
            x_refer_center = np.mean( x_refer, axis=0, keepdims=True )
            dist_train = euclidean_distances( x_refer, x_refer_center ).ravel()
            dist_target = euclidean_distances( x[:, feature_ids], x_refer_center ).ravel()
            rank = np.sum(dist_train<dist_target)+1   # rank = [2, n+1]
            weights.append( (len(x_refer)-rank+2) / (len(x_refer)+1) )

    weights = weights / np.sum(weights) # 归一化
    return weights

def get_des_weights_fast(x_test: np.ndarray, 
                         y_preds: list,
                         tree_features_list: List,
                         x_train: np.ndarray,
                         y_train: np.ndarray,
    ):
    """获取Dynamic Ensemble Selection的权重, 使用np.apply_along_axis() 加速

    Parameters
    ----------
    x_test: unknown data (#samples, #full_features)
    y_preds: shape of (#samples, #trees), 每个森林对每个样本的预测结果
    tree_features_list: List, 每棵树使用的特征的索引
    x_train: 训练集特征
    y_train: 训练集标签
    
    Returns
    -------
    weights: ndarray, shape of (#samples, #trees)
    
    """
    def anonymous(sample_id):
        sample_id = sample_id[0]
        x = x_test[sample_id]
        x = np.atleast_2d(x)
        y_pred = y_preds[sample_id]
        
        weights = []
        for tree_id, feature_ids in enumerate(tree_features_list):
            x_refer = x_train[ y_train == y_pred[tree_id], : ][ :, feature_ids ]
            if ( len(x_refer) == 0 or len(feature_ids)==0 ):
                weights.append(1/len(x_train))
            else:
                x_refer_center = np.mean( x_refer, axis=0, keepdims=True )
                dist_train = euclidean_distances( x_refer, x_refer_center ).ravel()
                dist_target = euclidean_distances( x[:, feature_ids], x_refer_center ).ravel()
                rank = np.sum(dist_train<dist_target)+1   # rank = [2, n+1]
                weights.append( (len(x_refer)-rank+2) / (len(x_refer)+1) )

        weights = weights / np.sum(weights) # 归一化

        return weights
    
    x_train = np.atleast_2d(x_train)
    weights = np.apply_along_axis(
        anonymous,
        axis=1,
        arr = np.arange(len(x_test), dtype=int)[:, None],
    )
    return weights.squeeze()

def selection(x, y):
    """基于特征标签相关性对作特征选择"""
    
    # 独立样本t检验
    t_test_results = []
    for i in range(x.shape[1]):
        cat1 = x[y==0, i]
        cat2 = x[y==1, i]
        t_test_results.append(stats.ttest_ind(cat1, cat2))

    # 方差分析
    anova_results = []
    for i in range(x.shape[1]):
        cat1 = x[y==0, i]
        cat2 = x[y==1, i]
        anova_results.append(stats.f_oneway(cat1, cat2))

    return t_test_results, anova_results

def feature_selection_ttest(x, y, alpha=0.05):
    """
    使用独立样本t检验进行特征选择。

    参数：
    ----
    x : array-like, shape (n_samples, n_features)
        特征矩阵
    y : array-like, shape (n_samples,)
        标签数组
    alpha : float
        显著性水平阈值，默认为0.05

    返回：
    -----
    significant_features : list
        通过t检验的显著性特征的索引列表
    """
    from scipy import stats

    significant_features = []
    for i in range(x.shape[1]):
        cat1 = x[y==0, i]
        cat2 = x[y==1, i]
        t_stat, p_val = stats.ttest_ind(cat1, cat2)
        if p_val < alpha:
            significant_features.append(i)
    return significant_features