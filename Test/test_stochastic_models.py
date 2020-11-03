import unittest

from stochastic_models import *

class TestBaseModel(unittest.TestCase):

    def test_parallel_S(self):
        p = StochasticModelBase(4,3,[1/12,9/12])
        p._generate_driver_samples()
        p._calculate_S()

        X  = p.X
        N,d,T = X.shape
        S = np.ones((N,d,T+1))
        for j in range(N):
            for i in range(d):
                for t in range(T):
                    delta = p.delta_T[t]
                    S[j,i,t+1] = S[j,i,t]*np.exp(0.2*X[j,i,t]*np.sqrt(delta)-0.04/2*delta)
        
        self.assertAlmostEqual(np.sum(S-p.S),0)