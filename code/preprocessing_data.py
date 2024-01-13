import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Iterator, Dict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.datasets import make_blobs

def load_simulation_multiview_data(random_state=None) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """模拟简单数据集"""
    n_samples = 200
    n_feature_list = [10, 5, 8]     
    n_class = 2
    n_view = len(n_feature_list)

    feature_dict = {}
    np.random.seed(random_state)
    labels = np.random.randint(n_class, size=n_samples)
    for v, n_features in enumerate(n_feature_list):
        _, sample_counts = np.unique(labels, return_counts=True)
        feature_dict[v], label = make_blobs(n_samples=sample_counts,
                                             n_features=n_features,
                                             shuffle=False,
                                             random_state=random_state,)
        # print(label)
    return feature_dict, labels

def load_simulation_multiview_data_cv(random_state=None) -> Tuple[Dict[int, np.ndarray], np.ndarray, List]:
    """模拟简单数据集, 交叉验证"""
    feature_dict, labels = load_simulation_multiview_data()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)
    cv_list = [(t,v) for (t,v) in rskf.split(feature_dict[0], labels)]
    return feature_dict, labels, cv_list


def load_data_to_df(file_path) -> pd.DataFrame:
    """读取数据集文件"""
    feature = pd.read_csv(file_path, index_col=0)
    return feature

def load_multiview_data(type:str) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """将所有views的data合并到一起, 导出到DataFrame
    Parameters
    ----------
    type: str, 'sub' or 'all'
        type='sub': 使用预实验数据
        type='all': 使用完整数据集

    Returns
    -------
    (feature, label): (DataFrame, DataFrame)

    """

    # if Type=="all", 导入所有数据(dataset_0403 in 43 server)
    if type == "all":
        feature_dict, label = load_toxric()
    return (feature_dict, label)

def load_combine_data_to_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """将所有views的data合并到一起, 导出到DataFrame
        预实验的数据
    
    Returns
    -------
    (feature, label): (DataFrame, DataFrame)

    """
    feature_dict, label = load_toxric()
    feature = pd.concat(feature_dict.values(), axis=1)
    # # feature = np.empty(shape=(0,0))
    # feature = pd.DataFrame(np.empty(shape=(0,0)))
    # for file in feature_file_list:
    #     feature = pd.concat([feature, load_data_to_df(os.path.join(dir_path, file))], axis=1)
    # label = load_data_to_df(os.path.join(dir_path, labelfile))
    return (feature, label)

def split_multiview_data(x_dict: Dict[int, np.ndarray], y, cv=None) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """为multi-view数据划分训练, 测试集"""
    if cv is None:
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=666)
        cv = [(t,v) for (t,v) in rskf.split(x_dict[0], y)]
    y = y.squeeze()
    train_idx, test_idx = cv[0]
    x_train_dict = {i:item[train_idx].copy() for i, item in enumerate(x_dict.values())}
    x_test_dict = {i:item[test_idx].copy() for i, item in enumerate(x_dict.values())}
    y_train = y[train_idx]
    y_test = y[test_idx]
    return x_train_dict, x_test_dict, y_train, y_test

def split_multiview_data_cv(x_dict: Dict[int, np.ndarray], y, cv=None, random_state=None) -> Iterator[Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray, np.ndarray]]:
    """为multi-view数据划分用于交叉验证的训练, 测试集"""
    if cv is None:
        if random_state is None: random_state=666
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)
        cv = [(t,v) for (t,v) in rskf.split(x_dict[0], y)]
    y = y.squeeze()
    for train_idx, test_idx in cv:
        x_train_dict = {i:item[train_idx].copy() for i, item in enumerate(x_dict.values())}
        x_test_dict = {i:item[test_idx].copy() for i, item in enumerate(x_dict.values())}
        y_train = y[train_idx]
        y_test = y[test_idx]
        yield x_train_dict, x_test_dict, y_train, y_test

    
def split_combining_data_cv(x, y, cv=None) -> Iterator:
    """为单view的模型划分训练集和测试集"""
    if cv is None:
        if random_state is None: random_state=666
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=random_state)
        cv = [(t,v) for (t,v) in rskf.split(x, y)]
    for train_idx, test_idx in cv:
        x_train = x[train_idx]
        x_test = x[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        yield x_train, x_test, y_train, y_test

################################## 从原始数据文件读取 ######################################
def load_toxric() -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """
    加载 TOXRIC 数据集(多模态)

    Hepatotoxicity.csv: 
        TAID: column name, 药物id
        Hepatotoxicity: column name, 标签
    Feature.csv:
        TAID: column name, 药物ID
    
    Returns
    --------
    feature_dict: Dict[int, DataFrame], 
        feature: DataFrame of shape (#samples, #features)
        特征字典, 存储多个view的特征
    label: DataFrame, shape of (#samples, )
        标签值
    """
    label_file = "/data/tq/dataset/toxric/dataset_0403/labels/Hepatotoxicity.csv"
    feature_dir = "/data/tq/dataset/toxric/dataset_0403/features/"
    feature_file_names = ["LINCS_L7_978.csv", "Morgan Fingerprint.csv", "Pubchem Fingerprint.csv", "target_matrix.csv"]
    label_df = pd.read_csv(label_file, index_col=0)["Hepatotoxicity"]
    # 对齐索引
    index = label_df.index
    for filename in feature_file_names:
        feature_file = os.path.join(feature_dir, filename)
        feature_index = pd.read_csv(feature_file, index_col=0).index
        index = np.intersect1d(index, feature_index)
    # # 保存索引映射文件
    # pd.DataFrame([np.arange(len(index)), index], col_name=["index", "TAID"]).to_csv("/data/tq/dataset/toxric/dataset_0403/TAID_Index_map.csv")

    # 读取数据
    feature_dict = {}
    for i, filename in enumerate(feature_file_names):
        feature_file = os.path.join(feature_dir, filename)
        feature_df = pd.read_csv(feature_file, index_col=0).loc[index, :]
        feature_dict[i] = feature_df
    label = label_df.loc[index]
    print(f"feature shapes: {[item.shape for item in feature_dict.values()]}")
    print(f"labels shapes: {label.shape}")
    return feature_dict, label

def load_toxric_combination_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载 TOXRIC 的合并数据集, 所有view都拼接到一起

    Returns
    -------
    feature: pd.DataFrame
    label_df: pd.DataFrame
    """
    feature_list, label = load_toxric()
    feature = pd.concat(feature_list, axis=1)
    return (feature, label)

def make_noise_data(x_dict: dict, mu:float=0., sigma:float=1.0, rate:float=1.):
    """构造带有噪声的数据集
    
    Parameters
    ----------
    mu: float, default 0, 高斯噪声的均值
    sigma: float, default 1.0, 高斯噪声的方差
    rate: 添加噪声的样本量
    """ 
    x_noise_dict = {}   
    n_samples = len(x_dict[0])
    n_added = int(rate*n_samples)
    idx_selected = np.random.randint(0, n_samples, n_added)
    for vi, x in x_dict.items():
        n_features = x.shape[1]
        noise = sigma * np.random.randn(n_added, n_features) + mu
        x[idx_selected] = x[idx_selected] + noise
        x_noise_dict[vi] = x
    return x_noise_dict

if __name__ == "__main__":
    # load_simulation_multiview_data(42)
    import joblib
    features_dict, labels = load_toxric()
    for v, df in features_dict.items():
        features_dict[v] = df.values
    labels = labels.values
    # 缓存特征_标签数据
    with open("/data/tq/dataset/toxric/dataset_0403/cache/features_dict_labels_0613.pkl", "wb") as handle:
        joblib.dump((features_dict, labels), handle)

    # 缓存 数据集划分索引
    rsfk = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=666)
    features_df, labels_df = load_toxric_combination_data()
    idx_train_list, idx_test_list = [], []
    for train_idx, test_idx in rsfk.split(features_df.values, labels_df.values):
        idx_train_list.append(train_idx)
        idx_test_list.append(test_idx)
    np.savetxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", 
               idx_train_list,
               fmt='%d')
    np.savetxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", 
               idx_test_list,
               fmt='%d')

    pass

    
    

    