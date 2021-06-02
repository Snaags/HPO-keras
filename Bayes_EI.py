from HPO_utils import config_space, Dispatcher
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import time

import numpy as np
from acquisition_functions import expected_improvement
from acquisition_functions import propose_location

def hp_dict_to_list(results:list, X , y) -> list:

    hold = []
    for i in results:
        y.append(i[1])
        for x in i[0]:
            hold.append(i[0][x])
        hold = remove_catagorical(hold)
        X.append(hold)
        hold = []

    return X, y


def remove_catagorical(X):
    catagorical_list = [0, 10, 13, 16, 19, 22]
    catagorical_list.reverse()

    for i in catagorical_list:
        X.pop(i)
    return X

def add_catagorical(X):

    X = list(X)

    catagorical_list = [0, 10, 13, 16, 19, 22]

    X.insert(0,"Adagrad")
    for i in catagorical_list[1:]:
        X.insert(i,"ReLU")
    return X



params_full = [
    "optimizer",
    "lr",
    "lr_decay",
    "weight_decay",
    "lambd",
    "momentum",
    "batch_size",
    "epochs",
    "num_layers",
    "hidden_layer_1",
    "activation_function_1",
    "dropout_1",
    "hidden_layer_2",
    "activation_function_2",
    "dropout_2",
    "hidden_layer_3",
    "activation_function_3",
    "dropout_3",
    "hidden_layer_4",
    "activation_function_4",
    "dropout_4",
    "hidden_layer_5",
    "activation_function_5",
    "dropout_5"
    ]
_n_ = [
    "lr",
    "lr_decay",
    "weight_decay",
    "lambd",
    "momentum",
    "num_layers",
    "hidden_layer_1",
    "hidden_layer_2",
    "hidden_layer_3",
    "hidden_layer_4",
    ]

import copy
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import WhiteKernel
def plot_gp(gp, x,X,y):
    mylist = []

    hold = np.asarray(X[0])
    for i in x:
        hold[8] = i
        mylist.append(hold.copy())


    mylist = np.asarray(mylist)

    X = np.asarray(X)

    y_pred, sigma = gp.predict(mylist, return_std=True)
    plt.figure()
    plt.plot(X[:,8], y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    plt.savefig("BO_log_TEPS_new_kernel.png")


import csv

def load_evaluations(filename):
    x = []

    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):
            hold =[]
            hold.append(eval(row[0]))
            hold.append(float(row[1]))
            x.append(hold)
    return x

from Thompson_Sampling import TS
from sklearn.gaussian_process.kernels import WhiteKernel


import os 

from functools import partial
import scipy.optimize

class GPR_Large_Iter(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=15000, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        def new_optimizer(obj_func, initial_theta, bounds):
            res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options = {"maxiter":self._max_iter})
            return res.x, res.fun

        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)




class Int_Matern(Matern):
    """
    adds an int variable list that should contain truthy elements declaring if a dim is an integer of not

    """
    def __init__(self,int_variables,length_scale=1.0, length_scale_bounds=(1e-5, 1e5),nu=1.5):

        self.int_variables = tuple(int_variables)
        self.non_int_variables = tuple(1 - np.array(self.int_variables))
        super().__init__(length_scale, length_scale_bounds,nu)

    def _T(self,X):

        return np.round(X*self.int_variables) + X*self.non_int_variables

    def __call__(self, X, Y=None, eval_gradient=False):
        return super().__call__(self._T(X) , Y , eval_gradient)

def BO(train_function, configspace, kernel = "Matern"):
    import eval(train_function) as train_function
    
    int_list = [0,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,1,0]
    kernel = Int_Matern(int_variables = int_list,length_scale_bounds=(1e-10, 100000.0)) 
    GP = GPR_Large_Iter(kernel=kernel,normalize_y=True,n_restarts_optimizer=100)

    CS = config_space()
    bounds = CS.get_bounds()
    ts = TS(bounds,1000)
    init_size = 2
    configs = []
    max_iter = 2000
    current_iter = 1
    ##Random Initialisation
    best_acc = 0
    best_hp = 0

    csv_columns = ['Hyperparameters','Accuracy']
    csv_file = "BO_log_TEPS_new_kernel_fixed.csv"

    X = []
    y = []

    x_lin = np.linspace(2 ,200,1000)
    configs = [CS.build_random_config()] #Random initial config
    filename = "BO_log_TEPS_new_kernel_fixed.csv"
    if os.path.exists(filename):
        results = load_evaluations(filename)
        X,y = hp_dict_to_list(results,X,y)
        GP = GP.fit(np.asarray(X),y)
        plot_gp(GP,x_lin,X,y)
        config = propose_location(expected_improvement,GP,bounds,y,np.asarray(X))
        config = add_catagorical(config)
        config = CS.build_config_dict(config)

        config = [CS.make_valid(config[0])]


    else:
        configs = [CS.build_random_config()]
        while len(configs) < init_size:
            configs.append(CS.build_random_config())

    while current_iter < max_iter:

        results = Dispatcher(config_list,1,validation_flag=False,train_func=train_function)
        with open(csv_file, 'a') as csvfile:
            writer = csv.writer(csvfile)
            for data in results:
                writer.writerow(data)


        X,y = hp_dict_to_list(results,X,y)
        #X = remove_catagorical(X)

        GP = GP.fit(np.asarray(X),y)

        plot_gp(GP,x_lin,X,y)

        #configs = ts.get_samples(n_samples = 16,gp =GP)
        config = 0
        config = propose_location(expected_improvement,GP,bounds,y,np.asarray(X))


        configs_new = []

        config = add_catagorical(config)

        config = CS.build_config_dict(config)

        config = [CS.make_valid(config[0])]


    

if __name__ == '__main__':
    BO()
