class StochasticModelBase:
    def __init__(self):
        pass

    def __reset_samples(self):
        pass

    def generate_samples(self):
        pass

    def _generate_driver_samples(self):
        pass
    
    def _calculate_y(self):
        pass

    def generate_true_V(self):
        pass

    def _calculate_S(self):
        pass
    
    def _calculate_f(self):
        raise NotImplementedError



class MinPutStochasticModel(StochasticModelBase):
    def __init__(self):
        super().__init__()
    
    def _calculate_f(self):
        pass
