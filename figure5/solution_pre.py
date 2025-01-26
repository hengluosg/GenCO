from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
from scipy.stats import wasserstein_distance
import torch
import torch.nn as nn
import torch.optim as optim
import time
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import QuantileRegressor
from scipy.interpolate import interp1d
plt.rcParams['figure.max_open_warning'] = 150  # 设置警告阈值为 50
from torch.utils.data import DataLoader, TensorDataset

def solver(x):
    num_initial_points = 100* theta_dim  
    initial_points = np.random.uniform(lowbound, upbound, size=(num_initial_points,theta_dim ))  # 随机采样初始点
    best_solution = None
    best_value = float('inf')
    for theta0 in initial_points:
        result = minimize(objective_function, theta0, args=(x,), method='L-BFGS-B') 
        if result.fun < best_value:
            best_value = result.fun
            best_solution = result.x
    return best_solution,best_value


def objective_function(theta, x):
    theta, x = theta.reshape(1,theta_dim),x.reshape(covariate_dim,1)
    hat_x = A @ x
    f = np.sum(np.cos(2 * theta) + hat_x * np.sin(2 * theta) + 0.1 * theta**2)
    return f
def grid_sample( d, lowbound, upbound, point):
        n = int(round(point ** (1.0 / d))) +1 
        grid_1d = np.linspace(lowbound + 0.1, upbound - 0.1, n)
        grid_points = np.array(list(product(grid_1d, repeat=d)))
        if len(grid_points) > point:
            indices = np.random.choice(len(grid_points), size=point, replace=False)
            grid_points = grid_points[indices]
        return grid_points
    
    
if __name__ == '__main__':
    np.random.seed(42) 
    std = 1
    lowbound = -5
    upbound = 5
    eta = 0.01 
    n = 1000
    m = 100 
    K = 25
    covariate_dim1 = [2,3,4,6,8]
    theta_dim1 = [2,3,4,4,4]
    for t,theta_dim in enumerate(theta_dim1):
        
        print("theta_dim",theta_dim)
        theta_dim = theta_dim1[t]
        covariate_dim = covariate_dim1[t]
        A = np.full((theta_dim, covariate_dim), 1)
        
        num_test_points = 1000
        optimal = np.zeros((num_test_points, theta_dim))
        test_points = np.random.uniform(low=lowbound, high=upbound, size=(num_test_points, covariate_dim))
        for i,x_new in enumerate(test_points):
            if i % 100 == 0:
                print(i)
            x_new = x_new.reshape(1, covariate_dim)
            optimal[i,:] = solver(x_new)[0]
        np.save('covariates_SCENARIO_{}.npy'.format(t), test_points)
        np.save('solutions_SCENARIO_{}.npy'.format(t), optimal)
        
        # # np.save('theta4.npy', mse_method2_d)
        # # np.save('theta6.npy', mse_method3_d)

    
    
    
    
    
    
    
    
    
    
    
    


