import numpy as np




from gaussian_processes_util import plot_gp

###This is the layout of 
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


params_reduced = [
    "lr",
    "lr_decay",
    "weight_decay",
    "lambd",
    "momentum",
    "epochs",
    "num_layers",
    "hidden_layer_1",
    "dropout_1",
    "hidden_layer_2",
    "dropout_2",
    "hidden_layer_3",
    "dropout_3",
    "hidden_layer_4",
    "dropout_4",
    "hidden_layer_5",
    "dropout_5"
    ]
params = [
    "optimizer",
    "lr",
    "lr_decay",
    "weight_decay",
    "lambd",
    "momentum",
    "batch_size",
    "epochs",
    "num_layers",
    "layers"
    ]
from numpy.linalg import inv
class gaussian_process:
    def __init__(self,input_dims,noise = 1e-2,encodings = None):
        self.posterior_x = [] #This is a list of all evaluated configurations
        self.posterior_y = [] #This is the list of the configurations scores
        self.dim = input_dims
        self.encodings = dict()
        self.noise = noise
        self.cache_old = True
        self.cache = 0
        self.calls = 0

    def kernel(self,X1, X2, l=1.0, sigma_f=1.0):
        """
        Isotropic squared exponential kernel.
        
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).

        Returns:
            (m x n) matrix.
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

 

    def posterior(self,X_s, X_train, Y_train, l=20.0, sigma_f=30.0, sigma_y=1e-8):
        """
        Computes the suffifient statistics of the posterior distribution 
        from m training data X_train and Y_train and n new inputs X_s.
        
        Args:
            X_s: New input locations (n x d).
            X_train: Training locations (m x d).
            Y_train: Training targets (m x 1).
            l: Kernel length parameter.
            sigma_f: Kernel vertical variation parameter.
            sigma_y: Noise parameter.
        
        Returns:
            Posterior mean vector (n x d) and covariance matrix (n x n).
        """
        self.calls += 1
        print("number of prediction calls: ", self.calls)
        mu_prior = 50 
        c = 40
        if self.cache_old == True:
            K = self.kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
            self.cache = K
            self.cache_old = False
        else:
            K = self.cache
        K_s = self.kernel(X_train, X_s, l, sigma_f)
        K_ss = self.kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
        K_inv = inv(K)
        
        # Equation (7)
        mu_s = mu_prior +K_s.T.dot(K_inv).dot(Y_train -mu_prior)

        # Equation (8)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        
        return mu_s, cov_s

    def convert_hyperparameters(self, config):
        converted_config= []
        for i in params_reduced:

            if type(config[i]) ==list:
                for x in config[i]:
                    converted_config.append(x)
            else:
                converted_config.append(config[i])


        return converted_config

    def get_prediction(self,x_predict):
        x_out = []

        if type(x_predict) == list:
            num_samples = len(x_predict)
            for x in x_predict:
                print(x)
                x_out.append(self.convert_hyperparameters(x))
            x_predict =x_out
            x_predict,self.posterior_x,self.posterior_y = np.asarray(x_predict), np.asarray(self.posterior_x), np.asarray(self.posterior_y)
        
        else:
            x_predict = self.convert_hyperparameters(x_predict)
            x_predict,self.posterior_x,self.posterior_y = np.asarray(x_predict).reshape(1,-1), np.asarray(self.posterior_x), np.asarray(self.posterior_y)
        


        return self.posterior(x_predict,self.posterior_x,self.posterior_y,sigma_y = self.noise)

    def get_posterior_prediction(self):

        x_predict,self.posterior_x,self.posterior_y = np.asarray(self.posterior_x), np.asarray(self.posterior_x), np.asarray(self.posterior_y)

        return self.posterior(x_predict,self.posterior_x,self.posterior_y,sigma_y = self.noise)

    def update_posterior(self, new_x, new_y):
        self.cache_old = True
        converted_x = self.convert_hyperparameters(new_x)
        if type(self.posterior_x) == list:
            self.posterior_x.append(converted_x)
            self.posterior_y.append(new_y)
        else:
            print(self.posterior_y.shape,np.asarray(new_y).reshape(1,-1).shape)
            self.posterior_x = np.concatenate((self.posterior_x,np.asarray(converted_x).reshape(1,-1)),0)
            self.posterior_y = np.concatenate((self.posterior_y.reshape(-1,1),np.asarray(new_y).reshape(-1,1)),0)


  
    def get_dim(self):
        print(np.asarray(self.posterior_x).ndim)
        if np.asarray(self.posterior_x).ndim > 1:
            return np.asarray(self.posterior_x).shape[1]
        else:
            return np.asarray(self.posterior_x).shape[0]


    def get_prediction_af(self,x_predict):

            
        x_predict,self.posterior_x,self.posterior_y = np.asarray(x_predict).reshape(1,-1), np.asarray(self.posterior_x), np.asarray(self.posterior_y)
        


        return self.posterior(x_predict,self.posterior_x,self.posterior_y,sigma_y = self.noise)



        



