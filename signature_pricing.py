from signature_core import *
import numpy as np

def psi_derivative(u: float, t: float, psi: Sig, model_sig: Sig, model_sig_squared: Sig, rho: float, trunc: int) -> Sig:
	psi_1 = psi.projection_on( (0,) )
	psi_2 = psi.projection_on( (1,) )
	psi_22 = psi.projection_on( (1, 1) )

	a = 0.5 * psi_2.shuffle(psi_2, trunc=trunc)
	b = model_sig.shuffle(psi_22, trunc=trunc)* ( rho * 1j * u )
	c = psi_1
	d = model_sig_squared * (0.5 * (-u**2 - 1j * u))

	return a + b + c + d

def simulate_psi_euler(psi_0: Sig, u: float, maturity: float, model_sig: Sig, model_sig_squared, rk_subdivs: int, rho: float, trunc: int):
	psi = psi_0.copy()
	dt = maturity / rk_subdivs
	for i in range(rk_subdivs):
		psi = psi + dt * psi_derivative(u, i*dt, psi, model_sig, model_sig_squared, rho, trunc)

	return psi

def simulate_psi_rk4(psi_0: Sig, u: float, maturity: float, model_sig: Sig, model_sig_squared, rk_subdivs: int, rho: float, trunc: int):
	psi = psi_0.copy()
	dt = maturity / rk_subdivs
	for i in range(rk_subdivs):
		k1 = psi_derivative(u, i*dt, psi, model_sig, model_sig_squared, rho, trunc)
		k2 = psi_derivative(u, i*dt+dt/2, psi + 0.5*dt*k1, model_sig, model_sig_squared, rho, trunc)
		k3 = psi_derivative(u, i*dt+dt/2, psi + 0.5*dt*k2, model_sig, model_sig_squared, rho, trunc)
		k4 = psi_derivative(u, (i+1)*dt, psi + dt*k3, model_sig, model_sig_squared, rho, trunc)
		psi = psi + (dt/6)*(k1 + 2*k2 + 2*k3 +k4 )

	return psi

def european_call(initial_price: int, maturity: float, strike: float, model_signature: Sig, rho: float):
	"""
	Fair price of a european call option given the parameters, under a signature volatility model
	The model assumes that dS_t = S_t bracket(signature, brownian_signature_t) d (rho W_t + sqrt(1-rho**2) W^{ortho}_t)
	Params:
		initial_price: S0 price of the asset at t=0
		maturity: time to maturity
		strike: K strike price
		model_signature: a constant signature that defines the model
		rho: correlation used for the model
	"""

	trunc = 4
	rk_subdivs = 50
	integral_subdivs = 100
	k_0 = np.log( initial_price / strike )

	model_sig_squared = model_signature.shuffle(model_signature, trunc=trunc)
	psi0 = Sig([], 2, dtype=np.complex128)
	brownian_sig_0 = Sig([np.array(1)], 2, dtype=np.complex128) # Signature of a brownian at t=0

	dt = 20 / integral_subdivs

	integral_elements = np.zeros(integral_subdivs)
	for i in range(integral_subdivs):
		u = dt * i # TODO: change this, integral up to infinity instead, analyze the function and find better subdivisions?
		res = simulate_psi_euler(psi0, u-0.5j, maturity, model_signature, model_sig_squared, rk_subdivs, rho, trunc)
		characteristic_val = np.exp( bracket(res, brownian_sig_0) )
		integral_elements[i] = np.real( np.exp(1j*(u-0.5j)*k_0) * characteristic_val ) * dt / ( u**2 + 0.25 )
	
	return initial_price - strike / np.pi * 0.5*(integral_elements[1:] + integral_elements[:-1]).sum()