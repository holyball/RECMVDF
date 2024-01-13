'''
Description: 计算某个配置下的深度森林的平均生长深度
Author: tanqiong
Date: 2023-06-29 23:28:52
LastEditTime: 2023-06-29 23:28:56
LastEditors: tanqiong
'''
import pandas as pd
import numpy as np

def count_n_layers(dir:str, n_view:int):
    """收集每种view的组合的结果，并存到新的csv文件中"""
    from itertools import combinations
    import os 

    layers_counts = []
    for n_comb in range(2, n_view+1):
        for c in combinations(range(n_view), n_comb):
            # 视图特征组合
            str_name = str(c).replace(',', '_')
            result_path = dir+f"/{str_name}"
            for fold in range(25):
                filename = result_path + f"/fold_{fold}/MVUGCForest_info/sample_marks_predict_.csv"
                if not os.path.exists(result_path):
                    raise FileNotFoundError(result_path, "Not Found")
                # 打印结果存储的文件夹
                # print(filename)
                df = pd.read_csv(filename, index_col=0)
                n_layer = len(df.columns)
                layers_counts.append(n_layer)

    return np.mean(layers_counts), np.std(layers_counts)


if __name__ == "__main__":
    dir = [
        "/home/tq/uncertainty_estimation_0403/result/MVUGCForest_4views_apart_0618_WeightBoostrap",
        "/home/tq/uncertainty_estimation_0403/result/MVUGCForest_4views_apart_0619_AUROC",
         "/home/tq/uncertainty_estimation_0403/result/MVUGCForest_4views_apart_0619_AUPR",       
    ]
    for dir_name in dir:
        print(count_n_layers(dir_name, 4))