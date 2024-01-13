# 
# 不确定深度森林实验的评估方法
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from numpy import ndarray as arr
from typing import List, Tuple, Union
import os
import time
import logging
import sys

def reject(y_true: arr, opinion: arr, logger:logging.Logger =None):
    """按百分比拒绝不确定性高的预测结果"""
    index_sorted = np.argsort(opinion[:,-1])[::-1]    # uncertainty , 不确定度从大到小排序
    y_true = np.squeeze(y_true)
    y_pred = np.argmax(opinion[:,:-1], axis=1)
    if logger is None:
        logger = get_logger("rejector")
    logger.info(f"rejection: 0%\t{accuracy_score(y_true, y_pred):.4f}")
    for i in np.arange(0.1, 1., 0.1):
        mask = np.ones_like(y_true, dtype=bool)
        index = index_sorted[:int(index_sorted.shape[0]*i)].squeeze()
        mask[index] = False
        acc = accuracy_score(y_true[mask], y_pred[mask])
        logger.info(f"rejection: {i*100:.0f}%\taccuracy: {acc:.4f}")

def uncertainty_acc_curve(y_true: arr, opinion: arr, logger:logging.Logger =None):
    """绘制不确定度和对应的识别精度
    Returns
    -------
    map_df: DataFrame, the content of map_df is like:
            ========================================
                percent   uncertainty   accuracy
            ========================================
                10%         0.2         0.4
                20%         0.25        0.45
                ...
                100%        0.40        0.90
            ========================================

    """

    index_sorted = np.argsort(opinion[:,-1])        # uncertainty , 不确定度从小到大排序
    y_true = np.squeeze(y_true)
    y_pred = np.argmax(opinion[:,:-1], axis=1)
    opinion = opinion[index_sorted]
    y_true = y_true[index_sorted]
    y_pred = y_pred[index_sorted]
    if logger is None:
        logger = get_logger("rejector")
    logger.info(f"rejection: 0%\t{accuracy_score(y_true, y_pred):.4f}")
    percent_uncertainty_acc_map_list = []
    for i in np.arange(0.1, 1., 0.1):
        count = int(len(y_true)*i)
        acc = accuracy_score(y_true[:count], y_pred[:count])
        percent_uncertainty_acc_map_list.append([
            f"{i*100}%", opinion[count-1][-1], acc
        ])
        logger.info(f"rate: {i*100:.0f}%\t uncertainty: {opinion[count-1][-1]:.4f}\taccuracy: {acc:.4f}")
    
    map_df = pd.DataFrame(percent_uncertainty_acc_map_list, columns=['percent','uncertainty', 'accuracy'])
    return map_df
    

def save_opinions(y_true: arr, opinion: arr, dir: str):
    y_pred = np.argmax(opinion[:,:-1], axis=1).reshape((-1,1))
    res = np.concatenate([y_true.reshape((-1, 1)), y_pred, opinion], axis=1)
    res = pd.DataFrame(res)
    res.columns = ["y_true", "y_pred", "b", "d", "u"]
    curtime = time.localtime(time.time()) 
    curtime = time.strftime('%Y-%m-%d %H-%M-%S', curtime)
    filename = f"opinion_{curtime}.csv"
    res.to_csv(os.path.join(dir, filename))
    time.sleep(1)

def get_logger(name, file_path=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')
    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handle = logging.StreamHandler(sys.stderr)
    console_handle.setLevel(logging.INFO)
    console_handle.setFormatter(formatter)
    logger.addHandler(console_handle)

    return logger

def partition_idx(y_true: arr, opinion: arr) -> Tuple[arr]:
    """划分样本类型
    
    Returns
    -------
    easy_idx: 
    hard_idx:
    risky_idx:
    outlier_idx:
    """
    low_uncertainty = 0.1
    high_uncertainty = 0.4
    y_pred = np.argmax(opinion[:, 0:-1], axis=1)
    uncertainty = opinion[:, -1]
    correct_mask = y_pred == y_true
    error_mask = y_pred != y_true
    easy_idx = np.argwhere(np.logical_and(uncertainty<low_uncertainty, correct_mask)).squeeze()
    hard_idx = np.argwhere(np.logical_and(uncertainty>high_uncertainty,  error_mask)).squeeze()
    risky_idx = np.argwhere(np.logical_and(uncertainty>high_uncertainty, correct_mask)).squeeze()
    outlier_idx = np.argwhere(np.logical_and(uncertainty<low_uncertainty, error_mask)).squeeze()
    normal_mask = np.logical_and(uncertainty>low_uncertainty, uncertainty<high_uncertainty)
    normal_correct_idx = np.argwhere(np.logical_and(normal_mask, correct_mask))
    normal_error_idx = np.argwhere(np.logical_and(normal_mask, error_mask))
    return easy_idx, normal_correct_idx,  risky_idx, outlier_idx, normal_error_idx, hard_idx,  

def draw_heatmap(count_list):
    """绘制各类样本分布的热力图"""
    import seaborn as sns
    import matplotlib.pyplot as plt
    matrix = pd.DataFrame(index=['correct prediction', 'error prediction'], columns=['low uncertainty', 'normal uncertainty','high uncertainty'], dtype=float)
    matrix.loc['correct prediction', 'low uncertainty'] = count_list[0]
    matrix.loc['correct prediction', 'normal uncertainty'] = count_list[1]
    matrix.loc['correct prediction', 'high uncertainty'] = count_list[2]
    matrix.loc['error prediction', 'low uncertainty'] = count_list[3]
    matrix.loc['error prediction', 'normal uncertainty'] = count_list[4]
    matrix.loc['error prediction', 'high uncertainty'] = count_list[5]
    
    # matrix.astype(np.float64)
    sns.heatmap(data=matrix, 
                cmap=plt.get_cmap('Blues'),
                annot=True)
    plt.show()
    
def check_opinion(y_true: arr, opinion: arr):
    """检查opinion的可靠性"""
    idx_list = partition_idx(y_true, opinion)
    count_list = [len(item) for item in idx_list]
    draw_heatmap(count_list)
    print(count_list)

def read_opinions_from_txt(filepath):
    """从txt文件中读取opinions, txt文件必须仅包含一组opinions的内容"""
    opinions = []
    with open(filepath, 'r') as handle:
        for line in handle.readlines():
            line = line.split('[')[-1].split(']')[0]
            line = line.split(' ')
            while '' in line:
                line.remove('')
            opinions.append([float(item) for item in line])
    opinions = np.array(opinions)
    return opinions
    
def get_stage_matrix(opinion_list, y_true):
    """查看每层预测的情况
    
    Examples:
    pred_statement = get_stage_matrix(gc.layers_opinions_list_train, y_train)
    
    pred_statement: shape of (#samples, #layers)
    """
    n_class = len(np.unique(y_true))
    y_pred_train_list = [np.argmax(opinion[:,:n_class], axis=1) for opinion in opinion_list]
    pred_statement = [y_true==y_pred for y_pred in y_pred_train_list]
    return np.transpose(pred_statement)

if __name__ == "__main__":
    opinions_file = "MVugcForest_info\\testset_layer_2_opinion.txt"
    opinions = read_opinions_from_txt(opinions_file)    
    y_test = pd.read_csv("demo_dataset\\y_test.csv", index_col=0).values.squeeze()
    check_opinion(y_test, opinions)

    pass
