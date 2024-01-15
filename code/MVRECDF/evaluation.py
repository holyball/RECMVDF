'''
Description: 为所有的评估方法都加入了空洞观点的评估
    ! 观点是二阶不确定性, 不能使用一阶不确定性的评估指标进行评估(比如auroc, aupr等)
    可以使用基于预测值的指标做评估
Author: tanqiong
Date: 2023-05-19 17:11:22
LastEditTime: 2023-06-25 21:52:27
LastEditors: tanqiong
'''
from numpy import argmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, \
                            average_precision_score, confusion_matrix
import numpy as np

def accuracy_opinion(y_true, opinion) -> float:
    # 保留非空洞观点
    not_vague_idx = np.sum(opinion[:, :-1], axis=1)==0

    y_pred = np.argmax(opinion[:, :-1], axis=1)
    return accuracy_score(y_true[not_vague_idx], y_pred[not_vague_idx])

def f1_opinion(y_true, opinion, average):
    """
    Parameters
    ----------
    average: str, 必填参数"""
    not_vague_idx = np.sum(opinion[:, :-1], axis=1)!=0
    y_pred = np.argmax(opinion[:, :-1], axis=1)
    return f1_score(y_true[not_vague_idx], y_pred[not_vague_idx], average)


def accuracy(y_true,y_proba) -> float:

    not_vague_idx = np.sum(y_proba, axis=1)!=0

    y_pred = argmax(y_proba, axis=1) 
    return accuracy_score(y_true[not_vague_idx],y_pred[not_vague_idx])

def f1_binary(y_true,y_proba):
    not_vague_idx = np.sum(y_proba, axis=1)!=0

    y_pred = argmax(y_proba, axis=1)
    f1=f1_score(y_true[not_vague_idx],y_pred[not_vague_idx],average="binary")
    return f1

def f1_micro(y_true,y_proba):
    not_vague_idx = np.sum(y_proba, axis=1)!=0
    
    y_pred = argmax(y_proba, axis=1)
    f1=f1_score(y_true[not_vague_idx],y_pred[not_vague_idx],average="micro")
    return f1

def f1_macro(y_true,y_proba):
    not_vague_idx = np.sum(y_proba, axis=1)!=0

    y_pred = argmax(y_proba, axis=1)
    f1=f1_score(y_true[not_vague_idx],y_pred[not_vague_idx],average="macro")
    return f1

def auroc(y_true, y_proba):
        
    # 处理y_true只有一个类别的情况:
    labels = np.unique(y_true)
    if (len(labels)==1):
        if labels[0] == 0:
            y_true = np.append(y_true, 1)
            y_proba = np.vstack( [y_proba, [[0,1]] ] )
        elif labels[0]==1:
            y_true = np.append(y_true, 0)
            y_proba = np.vstack( [y_proba, [[1, 0]] ] )
        else:
            raise ValueError("Only Support Binary Classification now!")
        
    not_vague_idx = np.sum(y_proba, axis=1)!=0

    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, -1]
    try:
        auroc_score = roc_auc_score(y_true[not_vague_idx], y_proba[not_vague_idx], multi_class="ovr")
    except ValueError:
        print(y_proba[not_vague_idx])
    return auroc_score

def aupr(y_true, y_proba):
    # 处理y_true只有一个类别的情况:
    labels = np.unique(y_true)
    if (len(labels)==1):
        if labels[0] == 0:
            y_true = np.append(y_true, 1)
            y_proba = np.vstack( [y_proba, [[0,1]] ] )
        elif labels[0]==1:
            y_true = np.append(y_true, 0)
            y_proba = np.vstack( [y_proba, [[1, 0]] ] )
        else:
            raise ValueError("Only Support Binary Classification now!")
        
    not_vague_idx = np.sum(y_proba, axis=1)!=0
    if y_proba.shape[1] == 2:
        y_proba = y_proba[:, -1]
    aupr = average_precision_score(y_true[not_vague_idx], y_proba[not_vague_idx])
    return aupr


# 基于分布的损失函数
def onehot(arr):
    return np.eye(len(set(arr)))[arr]

def onehot_label(arr, K):
    return np.eye(np.max(K))[arr]

# define loss function
def KL(alpha, K):
    """
    Parameters
    ----------
    alpha: 
    K: int, n_class"""
    from scipy.special import gamma, digamma
    beta=np.ones((1,K))
    S_alpha = np.sum(alpha,axis=1,keepdims=True)
    S_beta = np.sum(beta, axis=1,keepdims=True)
    lnB = gamma(S_alpha) - np.sum(gamma(alpha),axis=1,keepdims=True)
    lnB_uni = np.sum(gamma(beta),axis=1,keepdims=True) - gamma(S_beta)
    
    dg0 = digamma(S_alpha)
    dg1 = digamma(alpha)
    
    kl = np.sum((alpha - beta)*(dg1-dg0),axis=1,keepdims=True) + lnB + lnB_uni
    return kl

def mse_loss(y_true, alpha, global_step, annealing_step=10, is_sum=True, K=2): 
    """loss base SL
    Parameters
    ----------
    y_true: ndarray of shape(#samples, ), y_true
    alpha: ndarray of shape (#samples, #classes), evidence+1
    global_step: int, current training epoch, 可以用layer/depth替代
    annealing_step: int, default 10.
    is_sum: bool, 是否返回总损失
    """
    global_step = float(global_step)
    global_step = 5 # 固定
    annealing_step = float(annealing_step)
    try:
        p = onehot(y_true).astype(int)
    except IndexError:
        print("unique of y_true: {}".format(np.unique(y_true)))
        p = onehot_label(y_true, 2).astype(int)

    K = K 
    S = np.sum(alpha, axis=1, keepdims=True) 
    E = alpha - 1
    m = alpha / S
    
    A =np.sum((p-m)**2, axis=1, keepdims=True) 
    B = np.sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True) 
    
    annealing_coef = min(1.0, global_step/annealing_step)
    
    alp = E*(1-p) + 1 
    C = annealing_coef * KL(alp, K=K)
    if is_sum:
        return np.sum((A + B) + C)
    else:
        return (A + B) + C

if __name__ == "__main__":
    # # p = np.array([[1,0],
    # #               [1,0],
    # #               [0,1],])
    # y_true = [1, 1, 0,]
    # alpha = np.log(np.array([[17,25],
    #                          [25,17],
    #                          [25,17],]))
    # # alpha *= 4
    # global_step = 5
    # print(mse_loss(y_true, alpha, global_step, is_sum=False))

    y = [1,1,1]
    print(len(set(y)))
    if len(set(y)) == 1:
        tmp = np.max()
    print(np.eye(1))
    print(np.eye(2))
    np.eye(len(set(y)))[y]
    onehot(y)

