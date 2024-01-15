from Correlation import *
from itertools import combinations
if __name__ == "__main__":
    dir = "/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/origin_features_corr"
    
    with open("./data_log/dl_train.pkl", "rb") as handle:
        dl_train:DataLoader = joblib.load(handle)

    n_samples = dl_train.n_origin_sample

    y = dl_train.origin_labels
    
    for view_id in range(4):    
        selector = SelectKBest(f_classif, k=10)
        x_origin = dl_train.origin_features[view_id]
        x_selected = selector.fit_transform(x_origin, y)
        support = selector.get_support(indices=True)
        
        matrix = np.hstack( [x_selected, y.reshape((-1,1)) ] )
        # 查看新交互的特征与标签的相关度
        col_name = np.array([ f"f_{i}" for i in support ])
        col_name = np.append(col_name, 'label')
        generate_corr_pic(matrix, dir + f"/origin_features_corr_view_{view_id}.png", col_name = col_name)
        generate_corr_matrix(matrix, dir + f"/origin_features_corr_view_{view_id}.png.csv", col_name)




