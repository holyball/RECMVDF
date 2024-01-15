from Correlation import *
from itertools import combinations
if __name__ == "__main__":
    with open("./data_log/dl_train_beam_search_filter.pkl", "rb") as handle:
        dl_train:DataLoader = joblib.load(handle)

    n_samples = dl_train.n_origin_sample
    x_boost_name_dict = [names for names in dl_train.boost_features_names.values()]

    x_boost_dict = {i:dl_train.features[i][:n_samples, dl_train.n_origin_features[i]:] for i in range(4)}
    x_boost_dict = {v:pd.DataFrame(boost, columns=x_boost_name_dict[v]) for v, boost in x_boost_dict.items()}

    # view_id 
    view_id: int = 0
    x_origin = dl_train.origin_features[view_id]
    x_boost = x_boost_dict[view_id]
    x_boost_name = x_boost_name_dict[view_id]
    y = dl_train.origin_labels
    
    # intra_col_layer1 = {
    #     0:["intra_0"],
    #     1:["intra_0", "intra_1"],
    #     2:["intra_0"],
    #     3:["intra_0", "intra_1", "intra_2"],
    #     3:[f"intra_{i}" for i in range(6)],
    # }
    ["intra_0" * "intra_1" - "intra_0" + "intra_2"]
    intra_col_layer1 = {}
    intra_layer1 = {v:x_boost_dict[v][col_names] for v,col_names in intra_col_layer1.items()}   # 交互的特征

    # from hiDF.intersection import inter_generation
    # intra_layer1 = {v:item.values for v, item in intra_layer1.items()}
    # inter_list = inter_generation(intra_layer1, 0)

    import itertools
    # 提取每个字典值的一个元素，如果值为空列表，则使用一个占位元素
    intra_col_layer1_list = [v if v else ["placeholder"] for v in intra_col_layer1.values()]

    # 生成所有可能的组合
    combinations = list(itertools.product(*intra_col_layer1_list))

    inter_list = []
    operators = ["add", "sub", "mul"]
    # operators = ["mul"]
    for comb in combinations:
        inter = None
        for v,col in enumerate(comb):
            if col == 'placeholder': continue
            if inter is None:
                inter = x_boost_dict[v][col]
            else:
                operator = np.random.choice(operators, size=1)
                if operator == "add":
                    inter = inter + x_boost_dict[v][col]
                elif operator == "sub":
                    inter = inter - x_boost_dict[v][col]
                elif operator == "mul":
                    inter = inter * x_boost_dict[v][col]
                else:
                    raise ValueError(f"Not define operator: {operator}" )
        if inter is not None:
            inter_list.append(inter)
    inter_list = np.transpose(inter_list)

    # 查看新交互的特征与标签的相关度
    col_name = np.array([f"new_inter_{i}" for i in range(inter_list.shape[1])])
    inter_list = np.hstack( [inter_list, y[:, np.newaxis]] )
    col_name = np.append(col_name, 'label')
    generate_corr_pic(inter_list, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_layer0_inter_new.png", col_name = col_name)
    generate_corr_matrix(inter_list, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_layer0_inter_new.csv", col_name)

    # 查看原来的intra特征与标签的相关度
    col_name = [f"{i}_view_1" for i in intra_col_layer1[1]] + [f"{i}_view_3" for i in intra_col_layer1[3]]
    intra = pd.concat([intra_layer1[1], intra_layer1[3]], axis = 1)
    intra["label"] = y[:, np.newaxis]
    col_name = np.append(col_name, 'label')
    generate_corr_pic(intra, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_layer0_intra_origin.png", col_name = col_name)
    generate_corr_matrix(intra, f"/home/tq/uncertainty_estimation_0403/MVUGCForest/data_log/corrcoef_layer0_intra_origin.csv", col_name)




