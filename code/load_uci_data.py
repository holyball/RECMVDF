'''
Description: 加载UCI数据集
Author: tanqiong
Date: 2023-05-15 21:21:00
LastEditTime: 2023-06-18 19:55:14
LastEditors: tanqiong
'''
import numpy as np
import pandas as pd

def load_my_data(dataset):
    ''' uci dataset: adult / letter / yeast data
    '''

    path = "/data/tq/dataset/uci/{}/".format(dataset)
    if dataset=='adult':
        
        train_data = np.loadtxt(path+'train.txt', skiprows=1 )  
        train_label = np.loadtxt(path+'label_train.txt')  
        test_data = np.loadtxt(path+'test.txt', skiprows=1 )  
        test_label = np.loadtxt(path+'label_test.txt')
        # 成年人数据集需要做处理
        temp = np.where(train_data[:,13]==41)
        train_data[temp[0][0], 13] = 0
        
    if dataset == 'arrhythmia':
        ### GOOD
        # print('\nLoading UCI "Arrhythmia Data Set" , classification, (452 * 279 )')
        data = np.genfromtxt(f'{path}/arrhythmia.data', delimiter=',',dtype=float, missing_values='?', filling_values=0.0 )
        X = np.copy( data[:, :-1] )
        y = np.copy( data[:,-1] )
        y = y.astype(np.int32)
        train_data = X
        train_label = y
        test_data = None
        test_label = None

    return [train_data,train_label,test_data,test_label]

def my_one_hot_numpy( X, X_test , feature_ids ):
    '''
    Parameters
    ----------
    X, X_test : 
        numpy 2D array   
    feature_ids: 
        list of int

    return
    ------
    X, X_test:
        numpy 2D array 
    '''
    import pandas as pd
    #print('my one hot')
    df_train = pd.DataFrame(X)
    df_test = pd.DataFrame(X_test)

    #print( df_train[:3] )
    for idx in feature_ids:
        df_train = pd.concat([df_train, pd.get_dummies(df_train[idx], prefix='{}_'.format(idx))], axis=1)
        df_train.drop([idx],axis=1, inplace=True)
        #print( df_train[:3] )
        df_test = pd.concat([df_test, pd.get_dummies(df_test[idx], prefix='{}_'.format(idx))], axis=1)
        df_test.drop([idx],axis=1, inplace=True)

    return df_train.values.astype(float), df_test.values.astype(float)

def process_uci_to_multiview(train_data, test_data, feature_idxs_views):
    """将UCI数据集转化为multiview形式的数据集
    feature_idx_views: List[List], shape=(#views,), 存储每个view的特征索引
    """
    x_train_dict = {}
    x_test_dict = {}

    for i, feature_idxs in enumerate(feature_idxs_views):
        x_train_dict[i] = train_data[:, feature_idxs]
        x_test_dict[i] = test_data[:, feature_idxs]
    return x_train_dict, x_test_dict