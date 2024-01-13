'''
Description: 收集MVUGCForest的中间信息
Author: tanqiong
Date: 2023-07-14 11:22:41
LastEditTime: 2023-07-15 10:48:29
LastEditors: tanqiong
'''
import pandas as pd
import numpy as np
import os

def record_prediction_result(stage:str, sample_ids:list, dirpath:str=None, model=None, ):
    """记录中间数据"""

    if stage == "train" :
        file_suffix = "train_" 
    elif stage == "predict":
        file_suffix = "predict_"
    dirpath = model.logger_path if model is not None else dirpath

    ######## 记录最后一层 每个模态 的opnion与标签值
    excel_reader = pd.ExcelFile(
        dirpath+f"/prediction_all_views_{file_suffix}.xlsx",
        engine='openpyxl',)
    n_layer = len(excel_reader.sheet_names)
    sheet_name=f'layer_{n_layer-1}'
    opinions_views = pd.read_excel(excel_reader, sheet_name=sheet_name, index_col=0, header=[0,1])
    opinions_views.index = sample_ids 
    excel_reader.close()

    ######## 记录每一层的opinions与标签值
    excel_reader = pd.ExcelFile(
        dirpath + f"/prediction_all_layer_{file_suffix}.xlsx",
        engine='openpyxl',
        )
    n_layer = len(excel_reader.sheet_names)
    sheet_name=f'layer_{n_layer-1}'
    opinions_final_layer = pd.read_excel(excel_reader, sheet_name=sheet_name, index_col=0, header=[0])
    opinions_final_layer.index = sample_ids 
    excel_reader.close()
    
    return opinions_views, opinions_final_layer


def collective_uncertainty(dir_list, cv_list):
    """收集uncertainty, novel_prediction"""
    opinions_views_stack, opinions_final_layer_stack = None, None   # 所有数据的opinions
    for dir, cv in  zip(dir_list, cv_list):
        test_sample_ids = cv[1]
        opinions_views, opinions_final_layer = record_prediction_result('predict', test_sample_ids, dir)
        if opinions_views_stack is None:
            opinions_views_stack = opinions_views
            opinions_final_layer_stack = opinions_final_layer
        else:
            opinions_views_stack =  pd.concat([opinions_views_stack, opinions_views], axis=0)
            opinions_final_layer_stack =  pd.concat([opinions_final_layer_stack, opinions_final_layer], axis=0)
    
    return opinions_views_stack.sort_index(), opinions_final_layer_stack.sort_index()

if __name__ == "__main__":
    path_prefix = "/home/tq/uncertainty_estimation_0403/result/MVUGCForest_4views_0714_AUPR"
    train_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_train_ids_0613.txt", dtype=np.int64)
    test_ids = np.loadtxt("/data/tq/dataset/toxric/dataset_0403/cache/toxric_test_ids_0613.txt", dtype=np.int64)
    cv_list_all = [(t, v) for t, v in zip(train_ids, test_ids)]
    
    opinions_views_mean, opinions_final_layer_mean = 0, 0
    for i in range(5):
        dir_list = [os.path.join(path_prefix, f"fold_{i}/finding_best/MVUGCForest_info") for i in range(5*i, 5*(i+1))]

        cv_list = cv_list_all[5*i:5*(i+1)]

        opinions_views_stack, opinions_final_layer_stack = collective_uncertainty(dir_list, cv_list)

        pd.DataFrame(opinions_views_stack).to_csv(
            os.path.join(path_prefix, f"collective_information_views_{i}.csv")
        )
        pd.DataFrame(opinions_final_layer_stack).to_csv(
            os.path.join(path_prefix, f"collective_information_final_layer_{i}.csv")
        )

        opinions_views_mean += opinions_views_stack 
        opinions_final_layer_mean += opinions_final_layer_stack
    opinions_views_mean/=5
    opinions_views_mean.to_csv(
        os.path.join(path_prefix, "collective_information_views_mean.csv")
    )
    opinions_final_layer_mean/=5
    opinions_final_layer_mean.to_csv(
        os.path.join(path_prefix, "collective_information_final_layer_mean.csv")
    )
