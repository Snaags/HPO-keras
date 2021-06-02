
from scipy.stats import norm
import numpy as np

def expected_improvement(X,gpr,y,x_sample, xi=0.01,calls = 0):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''

    mu, sigma = gpr.predict(X.reshape(1,-1),return_cov = True)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample = gpr.predict(x_sample)
    mu_sample_opt = max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei[0][0]


from scipy.optimize import minimize

class counter:
	def __init__(self):
		self.count = 0

	def inc(self):
		self.count+=1 
		print(self.count)

from functools import partial
def propose_location(acquisition,gpr, bounds, y,x_sample,n_restarts=25):
	'''
	Proposes the next sampling point by optimizing the acquisition function.

	Args:
	acquisition: Acquisition function.
	X_sample: Sample locations (n x d).
	Y_sample: Sample values (n x 1).
	gpr: A GaussianProcessRegressor fitted to samples.

	Returns:
	Location of the acquisition function maximum.
	'''
	dim = len(bounds)
	bounds = np.asarray(bounds)
	c = counter()
	
	min_val = 1
	min_x = None
	options_dict = {"maxiter" : 1 ,"disp" : True,"maxfun": 10,"eps": 1e-4}
	def min_obj(x_sample,X):
	# Minimization objective is the negative acquisition function
		out = -acquisition(X, gpr,y,x_sample)
		return out

	min_obj_noisy = partial(min_obj,x_sample)

	# Find the best optimum by starting from n_restart different random points.
	for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
		x0 = x0.reshape(-1)

		res = minimize(min_obj_noisy, x0=x0, bounds=bounds, method='L-BFGS-B')

		if res.fun < min_val:
			print(res.fun)
			print(res.x)
			min_val = res.fun
			min_x = res.x           

	return min_x.reshape(-1).tolist()
