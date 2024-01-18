from concurrent.futures import ProcessPoolExecutor

def fit_module(module, x, y, group_id):
    module.fit(x, y, group_id=group_id)
    return module

def evaluate_proba(module, x, y, eval_func):
    return module.evaluate(x, y, eval_func, return_proba=True)

def generate_boosting_features_func(module, x, group_id):
    """顶级函数, 用于多进程序列化

    Args:
        moudule (Node | View): _description_
        x (_type_): _description_
        group_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    return module.generate_boosting_features(x, group_id)

def predict_module(module, x, y):
    y_proba_node = module.predict_proba(x, y)
    return y_proba_node
