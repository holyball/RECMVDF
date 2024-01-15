from concurrent.futures import ProcessPoolExecutor

def fit_view(view, x, y):
    view.fit(x, y)
    return view

def fit_node(node, x, y, group_id):
    node.fit(x, y, group_id=group_id)
    return node

def evaluate_proba(view, x, y, eval_func):
    return view.evaluate(x, y, eval_func, return_proba=True)

def generate_boosting_features_func(moudule, x, group_id):
    """顶级函数, 用于多进程序列化

    Args:
        moudule (Node | View): _description_
        x (_type_): _description_
        group_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    return moudule.generate_boosting_features(x, group_id)

def predict_node(node, x, y):
    y_proba_node = node.predict_proba(x, y)
    return y_proba_node
