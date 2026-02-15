import numpy as np
import numpy.random as npr
from signature_core import *

def generate_brownian(time_grid, n_curves):
	"""
	Generate n_curves samples of standard brownians noises
    Args:
        time_grid: array or 1D-np.ndarray with time values ranging from t0 to t1, used as a basis for the browninan simulation
        n_curves: number of independent curves to be generated
    
    Returns:
        np.ndarray of size (n_curves, len(time_grid))
	"""

	return np.cumsum(np.concatenate(
		(np.zeros((n_curves, 1)), np.sqrt(time_grid[1:] - time_grid[:-1]) * npr.normal(0, 1, (n_curves, len(time_grid)-1)) ),
		axis=1
	), axis=1)

def generate_mrgbm(t0, t1, brownian_noises, x0, kappa, etha, theta, alpha):
	(n_curves, n_divs) = brownian_noises.shape
	brownian_diffs = (brownian_noises[:,1:] - brownian_noises[:,:-1])
	process = np.zeros( (n_curves, n_divs) )
	dt = (t1-t0)/(n_divs-1)
	process[:, 0] = x0
	for i in range(1, process.shape[1]):
		process[:, i] = process[:, i-1] + kappa * (theta - process[:, i-1]) * dt + (etha + alpha * process[:, i-1])*brownian_diffs[:, i-1]

	return process

def generate_ornstein_uhlenbeck(t0, t1, brownian_noises, x0, kappa, etha, theta):
	(n_curves, n_divs) = brownian_noises.shape
	brownian_diffs = (brownian_noises[:,1:] - brownian_noises[:,:-1])
	process = np.zeros( (n_curves, n_divs) )
	dt = (t1-t0)/(n_divs-1)
	process[:, 0] = x0
	for i in range(1, process.shape[1]):
		process[:, i] = process[:, i-1] + kappa * (theta - process[:, i-1]) * dt + etha*brownian_diffs[:, i-1]

	return process

def compute_ornstein_uhlenbeck_signature(x0, kappa, etha, theta, exp_order=5):
	left = Sig([ np.array(x0), np.array([kappa*theta, etha]) ], 2)
	# compute exp(-kappa 1)
	right = [np.array(1.0)]
	for n in range(1,exp_order):
		right.append( make_canonical_element(tuple(0 for _ in range(n)), 2) * np.power(-kappa, n))
	return left * Sig(right, 2)

def compute_ornstein_uhlenbeck_signature_time(t, x0, kappa, etha, theta, exp_order=5):
	assert exp_order >= 2
	# the scalar part of the signature
	scalar_part = Sig([theta + np.exp(-kappa*t)*(x0-theta), np.zeros(2)],2)
	# left hand side of the decomposition: exp(+kappa 1)
	expon = [np.array(1.0)]
	for n in range(1, exp_order):
		expon.append( make_canonical_element(tuple(0 for _ in range(n)), 2) * np.power(kappa, n))
	# right hand side of the decomposition: etha 2
	rhs = Sig([np.zeros( () ), np.array([0.0, etha*np.exp(-kappa*t)]) ])
	# final product
	return scalar_part + (Sig(expon) * rhs)

def generate_naive_ornstein_sig(time_grid, brownians, brownian_sig, x0, kappa, etha, theta, max_sig_order):
	ornsteins = generate_ornstein_uhlenbeck(time_grid.min(), time_grid.max(), brownians, x0, kappa, etha, theta)
	true_uhlenbeck_sig = compute_ornstein_uhlenbeck_signature(x0, kappa, etha, theta, max_sig_order)
	ornsteins_approx = []
	for sig_order in range(1, max_sig_order+1):
		# compute the ornstein process using the signature volatility model
		true_uhlenbeck_process_sig = bracket_with_process(true_uhlenbeck_sig, brownian_sig, sig_order+1)
		ornsteins_approx.append(true_uhlenbeck_process_sig)

	return (ornsteins, ornsteins_approx)

def generate_stable_ornstein_sig(time_grid, brownians, brownian_sig, x0, kappa, etha, theta, max_sig_order):
	ornsteins = generate_ornstein_uhlenbeck(time_grid.min(), time_grid.max(), brownians, x0, kappa, etha, theta)
	ornstein_approx = []
	uhlenbeck_sig_time = []
	for t in time_grid:
		uhlenbeck_sig_time.append(
			compute_ornstein_uhlenbeck_signature_time(t, x0, kappa, etha, theta, max_sig_order)
		)

	for sig_order in range(1, max_sig_order+1):
		# compute the ornstein process using the signature volatility model
		uhlenbeck_process_approx = np.zeros(time_grid.shape[0])
		for i in range(time_grid.shape[0]):
			brownian_sig_time = [b[..., i] for b in brownian_sig]
			uhlenbeck_process_approx[i] = bracket(uhlenbeck_sig_time[i][:sig_order+1], brownian_sig_time)
		ornstein_approx.append(uhlenbeck_process_approx)

	return (ornsteins, ornstein_approx)

def compute_mrgbm_signature(x0, kappa, etha, theta, alpha, exp_order=5):
	assert exp_order >= 2
	# left hand side ...
	lhs = Sig([np.array(x0), np.array([kappa*theta - (alpha*etha)/2, etha])], 2)

	exponent = np.array([-(kappa + (alpha**2)/2), alpha])
	rhs = Sig([ np.array(1.0), exponent], 2)

	# compute the appoximation of exp( -exponent )
	curr_num = exponent
	curr_den = 1
	for n in range(2, exp_order+1):
		curr_num = shuffle_product(exponent, curr_num)
		curr_num_sig = [ np.zeros( tuple(2 for _ in range(i)) ) for i in range(len(curr_num.shape)+1) ]
		curr_den *= n
		
		curr_num_sig[len(curr_num.shape)] = curr_num / curr_den
		rhs += Sig(curr_num_sig)

	return lhs * rhs	

def generate_naive_mrgbm_sig(time_grid, brownians, brownian_sig, x0, kappa, etha, theta, alpha, max_sig_order):
	mrgbms = generate_mrgbm(time_grid.min(), time_grid.max(), brownians, x0, kappa, etha, theta, alpha)
	true_mrgbm_sig = compute_mrgbm_signature(x0, kappa, etha, theta, alpha, max_sig_order)

	mrgbm_approx = []
	for sig_order in range(1, max_sig_order+1):
		# compute the mrgbm process using the signature volatility model
		true_mrgbm_process_sig = bracket_with_process(true_mrgbm_sig.data[:sig_order+1], brownian_sig)
		mrgbm_approx.append(true_mrgbm_process_sig)

	return (mrgbms, mrgbm_approx)

def compute_mrgbm_signature_times(time_grid, x0, kappa, etha, theta, alpha, lamb, exp_order=5):
	lhs = compute_mrgbm_signature(x0, kappa, etha, theta, alpha, exp_order)
	lhs += Sig([np.array(-theta)], 2)
	# compute exp(lamb 1)
	rhs = [np.array(1.0)]
	for n in range(1,exp_order):
		rhs.append( make_canonical_element(tuple(0 for _ in range(n)), 2) * np.power(lamb, n))
	
	prod = lhs.shuffle(Sig(rhs), trunc=exp_order)

	theta_scalar = Sig([np.array(theta)], 2)
	values = []
	for t in time_grid:
		rest = scalar_prod( prod,  np.exp(-lamb*t) )
		rest += theta_scalar
		values.append(rest)

	return values

def generate_stable_mrgbm_sig(time_grid, brownians, brownian_sig, x0, kappa, etha, theta, alpha, max_sig_order):
	lamb = kappa + (alpha**2)/2
	mrgbms = generate_mrgbm(time_grid.min(), time_grid.max(), brownians, x0, kappa, etha, theta, alpha)
	mrgbms_approx = []
	mrgbm_sig_time = compute_mrgbm_signature_times(time_grid, x0, kappa, etha, theta, alpha, lamb, max_sig_order)

	for sig_order in range(1, max_sig_order+1):
		# compute the ornstein process using the signature volatility model
		mrgbm_process_approx = np.zeros(time_grid.shape[0])
		for i in range(time_grid.shape[0]):
			brownian_sig_time = [b[..., i] for b in brownian_sig]
			mrgbm_process_approx[i] = bracket(mrgbm_sig_time[i].data[:sig_order+1], brownian_sig_time)
		mrgbms_approx.append(mrgbm_process_approx)

	return (mrgbms, mrgbms_approx)