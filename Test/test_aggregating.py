import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import unittest

from aggregating.gridsearch import evaluate_model, create_logger
from aggregating.models import SimpleBagger
from aggregating.utils import *
from stochastic_models import *


class TestUtils(unittest.TestCase):
    def test_flattener(self):
        N_train = 2
        d = 3
        T = 2
        s_train = MaxCallStochasticModel(N_train,d,[1/12,11/12])
        s_train.generate_samples()
        y_train = s_train.y
        X_train = s_train.X
        self.assertTrue(X_train.shape == (N_train,d,T))
        flat  = flatten_X(X_train)
        self.assertTrue(flat.shape == (N_train,d*T))
        # flatten d's first check
        self.assertAlmostEqual(X_train[0,2,1], flat[0,5])
    
    def test_v_generator(self):
        self.assertAlmostEqual(generate_V_0(100000,[1/12,11/12],1),0.079,2)
    
    def test_set_generator(self):
        generate_train_set(100,[1/12,11/12],3)

class TestGridSearch(unittest.TestCase):
    def test_gridsearch_w_o_V0_arg(self):
        N_train = 1000
        N_test = 2000
        d = 1
        T = 2 

        lambda_range = (N_train*1e-9 , N_train*1e-3)
        alpha_range = (8.3*1e-5, 0.83)
        length_scale = np.sort(1/np.sqrt((2*alpha_range[0], 2*alpha_range[1])))

        #kernel
        kernel = RBF(length_scale= (length_scale[0] + length_scale[1])/2, length_scale_bounds=length_scale) \
            + WhiteKernel(noise_level= (lambda_range[0] + lambda_range[1])/2 , noise_level_bounds=lambda_range)

        s_train = MaxCallStochasticModel(N_train,d,[1/12,11/12])
        s_train.generate_samples()
        y_train = s_train.y
        X_train = s_train.X
        model = SimpleBagger(0,0,GaussianProcessRegressor(kernel,copy_X_train=False))
        errors = evaluate_model(model,{'M':5, 'train_size_alpha':0.2},flatten_X(X_train),y_train,1, [1/12,11/12],2,N_test,MaxCallStochasticModel)
        self.assertTrue(len(errors) ==2)
        self.assertTrue(errors[0]>0)
        self.assertTrue(errors[0] < 0.4)

    def test_gridsearch_w_V0_arg(self):
        N_train = 1000
        N_test = 2000
        d = 2
        T = 2 

        lambda_range = (N_train*1e-9 , N_train*1e-3)
        alpha_range = (8.3*1e-5, 0.83)
        length_scale = np.sort(1/np.sqrt((2*alpha_range[0], 2*alpha_range[1])))

        #kernel
        kernel = RBF(length_scale= (length_scale[0] + length_scale[1])/2, length_scale_bounds=length_scale) \
            + WhiteKernel(noise_level= (lambda_range[0] + lambda_range[1])/2 , noise_level_bounds=lambda_range)

        s_train = MaxCallStochasticModel(N_train,d,[1/12,11/12])
        s_train.generate_samples()
        y_train = s_train.y
        X_train = s_train.X
        V_0 = s_train.generate_true_V(0)
        V_0= V_0.mean()
        model = SimpleBagger(0,0,GaussianProcessRegressor(kernel,copy_X_train=False))
        errors = evaluate_model(model,{'M':5, 'train_size_alpha':0.2},flatten_X(X_train),y_train,d, [1/12,11/12],2,N_test,MaxCallStochasticModel,V_0=V_0)
        self.assertTrue(len(errors) ==2)
        self.assertTrue(errors[0]>0)
        self.assertTrue(errors[0] < 0.8)

