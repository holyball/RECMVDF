'''
Description: 使用束搜索算法做基于运算符模态间特征交互
Author: tanqiong
Date: 2023-07-03 17:43:02
LastEditTime: 2023-07-05 16:12:58
LastEditors: tanqiong
'''
import itertools
import random
import numpy as np
import pandas as pd
from typing import Dict

def inter_operator_fit(intra_dict:Dict[int, np.ndarray], label:np.ndarray, random_state:int=None):
    """模态之间的特征交互, 使用运算符生成交互特征, fit(拟合)
    Parameters
    ----------
    x_dict: dict, 
    depth: int,
    label: ndarray, 标签
    operations: list, 交互运算符
    inter_orders: list, 交互顺序, shape of (n_combinations, )
        
        operations:
        [["sum", "sub", "mul"],
         ["sub", "mul", "sum"],
         ...,
         ["mul", "sub", "sub"]]
        
        inter_orders: 
        [[0, 1, 2, 3],
         [2, 1, 3, 0], 
         ...,
         [0, 3, 2, 1]]
        对于每一组可能的特征组合, operations给出了两个原特征交互的操作符,
        inter_order给出了模态特征的交互顺序. 


    Returns
    -------
    x_new: ndarray of shape(#samples, #new features)
    """

    intra_dict:Dict[int, pd.DataFrame] = {v:pd.DataFrame(intra) for v, intra in intra_dict.items()}
    intra_cols = { v:intra.columns.to_list() for v, intra in intra_dict.items()}
    n_view_to_intersection = np.sum([1 for col in intra_cols.values() if col!=[]]).astype(int)

    # 如果没有任何一个view存在intra-section特征, 则直接跳过
    # 这不应该成为一种合法的情况, 看看是否能在后续作改进
    # transform方法也同理
    if n_view_to_intersection == 0:
        inter_list = np.empty((intra_dict[0].shape[0], 0))
        return inter_list, [], []
    
    import itertools
    # 提取每个字典值的一个元素，如果值为空列表，则使用一个占位元素
    intra_cols_list = [v if v else ["placeholder"] for v in intra_cols.values()]
    # 生成所有可能的组合
    combinations = list(itertools.product(*intra_cols_list))
    inter_list = []

    operators = ["add", "sub", "mul"]

    # 随机选择操作符
    np.random.seed(random_state)
    operations = np.random.choice( 
        operators, 
        size=( len(combinations), n_view_to_intersection-1 ),
    )
    inter_orders = []
    view_id_legal = [view_id for view_id, col in enumerate(intra_cols_list) if col[0] != "placeholder" ]     # 合法的模态特征索引

    # 如果只有一个模态产出了模态内交互特征
    if n_view_to_intersection == 1:
        for v_id, col in enumerate(intra_cols_list):
            if col[0] != 'placeholder':
                inter_list = intra_dict[v_id].values
                operations = [ [] for _ in inter_list ]
                inter_orders = np.array( [ [v_id] for _ in inter_list ] )
                return inter_list, operations, inter_orders
                # break
                
    # 如果只有两个特征进行交互, 则寻找最优的符号
    elif n_view_to_intersection == 2:
        for i, comb in enumerate(combinations):

            best_corr = 0
            v1, v2 = view_id_legal
            col1, col2 = comb[v1], comb[v2]
            # 寻找最优操作符
            for operator in operators:
                f_new_tmp = operating(intra_dict[v1][col1], intra_dict[v2][col2], operator)
                cur_corr = np.corrcoef(f_new_tmp, label)[0,1]
                if np.abs(cur_corr) > np.abs(best_corr):
                    f_inter = f_new_tmp
                    best_corr = cur_corr
                    operations[i] = [operator]

            inter_list.append( f_inter )
            inter_orders.append( view_id_legal )
        operations = operations[:, 0][:, np.newaxis]

    # 如果交互特征数量大于2, 使用beam search方法, 固定operator, 寻找局部最优的特征交互排列
    else:
        for i, comb in enumerate(combinations):
            cur_order = []
            res_view_id = view_id_legal.copy()
            for k, operator in enumerate(operations[i]):

                # 如果是第一操作符
                if k == 0:
                    v_combs = list(itertools.combinations(res_view_id, 2))
                    best_corr = 0
                    best_view_id = None
                    # 寻找前两个模态的最优排列
                    for two_view in v_combs:
                        v1, v2 = two_view
                        col1, col2 = comb[v1], comb[v2]
                        f_new_tmp = operating(intra_dict[v1][col1], intra_dict[v2][col2], operator)
                        cur_corr = np.corrcoef(f_new_tmp, label)[0,1]
                        if np.abs(cur_corr) > np.abs(best_corr):
                            f_inter = f_new_tmp
                            best_corr = cur_corr
                            best_view_id = two_view
                    for view_id in best_view_id:
                        res_view_id.remove(view_id)
                        cur_order.append(view_id)
                # 如果不是第一操作符
                else:
                    f_inter_copy = f_inter.copy()   # 拷贝当前的交互特征, 防止后面更新的时候丢失原始f_inter
                    best_corr = 0
                    best_view_id = None
                    # 寻找剩下模态的最优排列
                    for view_id in res_view_id:
                        v = view_id
                        col = comb[v]
                        f_new_tmp = operating(f_inter_copy, intra_dict[v][col], operator)
                        cur_corr = np.corrcoef(f_new_tmp, label)[0,1]
                        if np.abs(cur_corr) > np.abs(best_corr):
                            f_inter = f_new_tmp
                            best_corr = cur_corr
                            best_view_id = view_id
                    res_view_id.remove(best_view_id)
                    cur_order.append(best_view_id)
            
            inter_orders.append(cur_order)
            inter_list.append(f_inter)

    inter_list = np.transpose(inter_list)
    if len(inter_list) == 0:
        raise ValueError(f"无法产生模态间交互特征, 因为不存在intra-section feature")
        # inter_list = np.empty((intra_dict[0].shape[0], 0))
    return inter_list, operations, inter_orders

def inter_operator_transform(intra_dict:Dict[int, np.ndarray], operations:list, inter_orders:list):
    """ 模态间特征交互 transform """

    intra_dict:Dict[int, pd.DataFrame] = {v:pd.DataFrame(intra) for v, intra in intra_dict.items()}
    intra_cols = { v:intra.columns.to_list() for v, intra in intra_dict.items()}
    n_view_to_intersection = np.sum([1 for col in intra_cols.values() if col!=[]]).astype(int)

    # 如果operations是空
    if len(operations) == 0 and len(inter_orders) == 0:
        inter_list = np.empty((intra_dict[0].shape[0], 0))
        return inter_list

    import itertools
    # 提取每个字典值的一个元素，如果值为空列表，则使用一个占位元素
    intra_cols_list = [v if v else ["placeholder"] for v in intra_cols.values()]
    # 生成所有可能的组合
    combinations = list(itertools.product(*intra_cols_list))
    inter_list = []
    
    # 如果只有一个模态产出了模态内交互特征
    if n_view_to_intersection == 1:
        for v_id, col in enumerate(intra_cols_list):
            if col[0] != 'placeholder':
                return intra_dict[v_id].values  # 直接返回

    # 如果交互了模态间特征
    else:
        for i, comb in enumerate(combinations):
            v = inter_orders[i][0]
            col = comb[v]
            f_inter = intra_dict[v][col]
            for v, operator in zip( inter_orders[i][1:], operations[i] ):
                col = comb[v]
                f_inter = operating(f_inter, intra_dict[v][col], operator)
            inter_list.append(f_inter)
    inter_list = np.transpose(inter_list)
    return inter_list


def operating(f1, f2, operator):
    """根据operator作特征交互"""
    if operator == "add":
        f_new = f1 + f2
    elif operator == "sub":
        f_new = f1 - f2
    elif operator == "mul":
        f_new = f1 * f2
    else:
        raise ValueError(f"Not define operator: {operator}" )
    
    # 防止生成无用特征
    if np.std(f_new) == 0.0:
        f_new[0] += 0.01
    return f_new

if __name__ == "__main__":
    n_features = [2, 0, 0, 0]
    n_sample = 20
    feature_dict = {}
    for i, n_feature in enumerate(n_features):
        feature_dict[i] = np.random.random(size=(n_sample, n_feature ))
    labels = np.zeros(n_sample)
    labels[ np.random.choice(np.arange(n_sample), size=10) ] = 1
    
    from copy import deepcopy
    feature_dict_copy = deepcopy(feature_dict)

    inter_list, operations, inter_orders = inter_operator_fit(feature_dict, labels)

    inter_list_transform = inter_operator_transform(feature_dict, operations, inter_orders)

    print ( np.all(np.array(inter_list_transform) == np.array(inter_list)) )
    print ( 0 )
    pass
