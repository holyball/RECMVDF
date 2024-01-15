class BaseLearner():
    def __init__(self, **kwargs) -> None:
        self.random_state = kwargs.get("random_state", None)
        self._is_fitted = False
    
    def fit(self):
        pass
    def evaluate(self):
        pass
    def predict_proba(self):
        pass
    def predict(self):
        pass