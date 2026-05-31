import signature_core as core
import numpy as np
import scipy

def psi_derivative(
		u: float, t: float, psi: core.Sig2D,
		model_sig: core.Sig2D, model_sig_squared: core.Sig2D,
		rho: float, trunc: int,
		sig_cache: core.ShuffleCache = None
	) -> core.Sig2D:

	psi_1 = core.projection_on(psi, 0b0, 1)
	psi_2 = core.projection_on(psi, 0b1, 1)
	
	res = core.shuffle(psi_2, psi_2, trunc, sig_cache)
	res *= 0.5

	b = core.shuffle(model_sig, psi_2, trunc, sig_cache)
	b *= ( rho * 1j * u )
	res += b

	d = model_sig_squared.copy()
	d *= (0.5 * (-u**2 - 1j * u))
	res += d

	e = core.projection_on(psi, 0b11, 2)
	e *= 0.5
	res += e

	res += psi_1

	return res

def simulate_psi_euler(psi_0: core.Sig2D, u: float, maturity: float, model_sig: core.Sig2D, model_sig_squared, rk_subdivs: int, rho: float, trunc: int, sig_cache: core.ShuffleCache = None):
	psi = psi_0.copy()
	dt = float(maturity / rk_subdivs)
	for i in range(rk_subdivs):
		deriv = psi_derivative(u, i*dt, psi, model_sig, model_sig_squared, rho, trunc, sig_cache)
		psi = psi + (dt * deriv)
	return psi

def simulate_psi_rk4(psi_0: core.Sig2D, u: float, maturity: float, model_sig: core.Sig2D, model_sig_squared, rk_subdivs: int, rho: float, trunc: int, sig_cache: core.ShuffleCache = None):
	psi = psi_0.copy()
	dt = float(maturity / rk_subdivs)
	for i in range(rk_subdivs):
		k1 = psi_derivative(u, i*dt, psi, model_sig, model_sig_squared, rho, trunc, sig_cache)
		k2 = psi_derivative(u, i*dt+dt/2, psi + (0.5*dt)*k1, model_sig, model_sig_squared, rho, trunc, sig_cache)
		k3 = psi_derivative(u, i*dt+dt/2, psi + (0.5*dt)*k2, model_sig, model_sig_squared, rho, trunc, sig_cache)
		k4 = psi_derivative(u, (i+1)*dt, psi + dt*k3, model_sig, model_sig_squared, rho, trunc, sig_cache)
		psi = psi + (dt/6)*(k1 + 2*k2 + 2*k3 +k4 )

	return psi

def european_call_bs(initial_price: float, maturity: float, strike: float, r, sigma):
    d1 = (np.log(initial_price / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    call_price = initial_price * scipy.stats.norm.cdf(d1) - strike * np.exp(-r * maturity) * scipy.stats.norm.cdf(d2)
    return call_price

def european_call_sig(
	initial_price: float,
	maturity: float,
	strike: float,
	model_signature: core.Sig2D,
	rho: float,
	trunc: int = 4,
	rk_subdivs = 50,
	integral_subdivs = 100,
	r_bs = None,
	vol_bs = None,
	cache = False
	):
	"""
	Fair price of a european call option given the parameters, under a signature volatility model
	The model assumes that dS_t = S_t bracket(signature, brownian_signature_t) d (rho W_t + sqrt(1-rho**2) W^{ortho}_t)
	Params:
		initial_price: S0 price of the asset at t=0
		maturity: time to maturity
		strike: K strike price
		model_signature: a constant signature that defines the model
		rho: correlation used for the model
		trunc:
		rk_subdivs:
		integral_subdivs:
		r_bs, vol_bs: Used for variance reduction, risk-free rate & vol
	"""

	if cache is None:
		cache = core.compute_shuffle_cache(trunc)

	k_0 = np.log( initial_price / strike )
	bs_call = None
	variance_reduction = (r_bs is not None) and (vol_bs is not None)
	if variance_reduction:
		bs_call = european_call_bs(initial_price, maturity, strike, r_bs, vol_bs)

	model_sig_squared = core.shuffle(model_signature, model_signature, trunc)
	psi0 = core.Sig2D(trunc, 0.0)
	upper_bound = 500

	# default integrand
	def integrand_vr(u):
		return core.european_call_integrand_vr(u, k_0, maturity, model_signature, model_sig_squared, rho, r_bs, vol_bs, trunc, rk_subdivs, upper_bound, cache)

	# integrand if the variance reduction is enabled
	def integrand(u):
		if u > upper_bound:
			return 0
		res = simulate_psi_rk4(psi0, u-0.5j, maturity, model_signature, model_sig_squared, rk_subdivs, rho, trunc, cache)
		characteristic_val = np.exp( res.get_element([]) )
		return np.real( np.exp(1j*(u-0.5j)*k_0) * characteristic_val ) / ( u**2 + 0.25 )

	if variance_reduction:
		integral, _ = scipy.integrate.quad(integrand_vr, 0, np.inf, limit=integral_subdivs, epsabs=0.1)
	else:
		integral, _ = scipy.integrate.quad(integrand, 0, np.inf, limit=integral_subdivs, epsabs=0.1)

	if variance_reduction:
		bs_call = european_call_bs(initial_price, maturity, strike, r_bs, vol_bs)
		return bs_call - strike / np.pi * integral
	else:
		return initial_price - strike / np.pi * integral
	
