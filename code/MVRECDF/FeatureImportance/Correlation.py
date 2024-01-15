'''
Description: 特征相关性
Author: tanqiong
Date: 2023-06-24 15:03:57
LastEditTime: 2023-07-25 16:31:20
LastEditors: tanqiong
'''
import sys
sys.path.append("/home/tq/uncertainty_estimation_0403/MVUGCForest")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import joblib
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2,f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from MVUGCForest import MVUGCForest
from FeatureImportance.mip import mip_score
from MVRECDF.dataloader import DataLoader
np.set_printoptions(precision=4)

def generate_corr_matrix(feature_matrix, save_path, col_name):
    """保存相关系数矩阵到csv文件"""
    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(feature_matrix, rowvar=False)

    # 将相关系数矩阵转换为DataFrame
    df_correlation = pd.DataFrame(correlation_matrix,columns=col_name, index=col_name)

    # 将相关系数矩阵保存到CSV文件
    df_correlation.to_csv(save_path)

    return df_correlation


def generate_corr_pic(feature_matrix, save_path=None, col_name=None):
    """生成相关系数热力图, 并保存到save_path"""

    # 计算相关系数矩阵
    correlation_matrix = np.corrcoef(feature_matrix, rowvar=False)

    # 创建相关性热力图
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                     vmin=-1, vmax=1,
                     xticklabels=range(feature_matrix.shape[1]),
                     yticklabels=range(feature_matrix.shape[1]))
    plt.title("Feature Correlation Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Index")

    # 设置标签
    if col_name is not None:
        angle = 45
        ax.set_xticklabels(col_name, rotation=angle)
        ax.set_yticklabels(col_name, rotation=angle)
        ax.yaxis.set_label_position('right')  # 将y轴标签放置在右侧
    # 调整图像布局
    plt.subplots_adjust(top=0.85)

    # 保存csv源数据文件
    csv_path = save_path+".csv"
    generate_corr_matrix(feature_matrix, csv_path, col_name)

    # 保存图片
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    with open("./data_log/dl_train.pkl", "rb") as handle:
        dl_train:DataLoader = joblib.load(handle)

    n_samples = dl_train.n_origin_sample
    x_boost_dict = [dl_train.features[i][:n_samples, dl_train.n_origin_features[i]:] for i in range(4)]
    x_boost_name_dict = [names for names in dl_train.boost_features_names.values()]

    # view_id 
    view_id: int = 0
    x_origin = dl_train.origin_features[view_id]
    x_boost = x_boost_dict[view_id]
    x_boost_name = x_boost_name_dict[view_id]
    y = dl_train.origin_labels

    # from MVUGCForest.util import selection
    # t_test_results, anova_results = selection(x_origin, y)
    # from sklearn.feature_selection import f_classif
    # anova_results_sklearn = f_classif(x_origin, y)

    for view_id in range(4):
        # view_id: int = 0
        x_origin = dl_train.origin_features[view_id]
        x_boost = x_boost_dict[view_id]
        x_boost_name = x_boost_name_dict[view_id]

        y = dl_train.origin_labels
        x_train, x_test, x_boost_train, x_boost_test, y_train, y_test =  train_test_split(x_origin, x_boost, y, random_state=6)
        x_boost_train_df= pd.DataFrame(x_boost_train)
        x_boost_train_df['label'] = y_train

        print("x boost feature shape: ", x_boost_train.shape)
        print("x origin feature shape: ", x_train.shape)
        
        # # 分析特征矩阵中的相关性
        # inter_boost = x_boost[:, np.char.find(x_boost_name, sub="inter") != -1]
        # col_name = np.array(x_boost_name)[ np.char.find(x_boost_name, sub="inter") != -1 ]
        # inter_boost = np.hstack( [inter_boost, y[:, np.newaxis]] )
        # col_name = np.append(col_name, 'label')
        # analyze_feature_correlation(inter_boost, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_view{view_id}_inter.png", col_name = col_name)
        # calculate_and_save_correlation_matrix(inter_boost, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_view{view_id}_inter.csv", col_name)

        # intra_boost = x_boost[:, np.char.find(x_boost_name, sub="intra") != -1]
        # col_name = np.array(x_boost_name)[ np.char.find(x_boost_name, sub="intra") != -1 ]
        # intra_boost = np.hstack( [intra_boost, y[:, np.newaxis]] )
        # col_name = np.append(col_name, 'label')
        # analyze_feature_correlation(intra_boost, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_view{view_id}_intra.png", col_name = col_name)
        # calculate_and_save_correlation_matrix(intra_boost, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_view{view_id}_intra.csv", col_name)

        # boost = x_boost[:, np.char.find(x_boost_name, sub="boost") != -1]
        # col_name = np.array(x_boost_name)[ np.char.find(x_boost_name, sub="boost") != -1 ]
        # boost = np.hstack( [boost, y[:, np.newaxis]] )
        # col_name = np.append(col_name, 'label')
        # analyze_feature_correlation(boost, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_view{view_id}_boost.png", col_name = col_name)
        # calculate_and_save_correlation_matrix(boost, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_view{view_id}_boost.csv", col_name)


