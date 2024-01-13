import numpy as np
from numpy import ndarray as arr
from typing import Union, Tuple, List

def arctanh(x: arr) -> arr:
    return np.arctanh(x)

def sigmoid(x:arr) -> arr:
    return np.exp(x)/(1+np.exp(x))

def logit(x: arr) -> arr:
    """sigmoid的反函数"""
    return np.log(x/(1-x))

def softmax(x:arr) -> arr:
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def ReLU(x: arr) -> arr:
    # return x * (x > 0)
    return np.where(x>0, x, 0)

def approx_step(x: arr, bias:float, k:int=10, ) -> arr:
    """阶跃函数的近似函数"""
    # if bias is None:
    #     bias = 1/len(np.unique(x))
    return 1 / (1 + np.exp(-2*k*(x-bias)))

def calculate_entropy(proba: arr) -> arr:
    """
    计算给定概率向量的熵值。
    
    Parameters
    ----------
    proba: np.ndarray
    """
    # 确保概率向量的和为1
    if np.any(np.abs(np.sum(proba, axis=1) - 1) > 1e-10):
        raise ValueError("输入的概率向量不是有效的概率分布")
    
    # 将概率向量中的0处理为1e-10, 1处理为1-(1e-10)
    minimum = 1e-10
    proba = np.where(proba==0, minimum, proba)
    proba = np.where(proba==1, 1-minimum, proba)
    # 计算类的数量
    n_class = proba.shape[1]
    #计算熵
    entropy_mat = -np.array([proba[:, i] * np.log2(proba[:,i]) for i in range(n_class)]).T
    entropy = np.sum(entropy_mat, axis=1)
    entropy = np.where(entropy<1e-7, 0, entropy)
    return entropy
# if __name__ == "__main__":
#     a = np.linspace(0,1,11)
#     print(a)
#     print(approx_step(a, bias=0.5, k=5))