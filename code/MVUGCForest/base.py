class BaseLearner():
    def __init__(self, **kwargs) -> None:
        self.random_state = kwargs.get("random_state", None)
        