import signature_core as core
import numpy as np
import scipy

def european_call_bs(initial_price: float, maturity: float, strike: float, r: float, sigma: float) -> float:
    d1 = (np.log(initial_price / strike) + (r + 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)
    return initial_price * scipy.stats.norm.cdf(d1) - strike * np.exp(-r * maturity) * scipy.stats.norm.cdf(d2)

def european_call_sig(
    initial_price: float,
    maturity: float,
    strike: float,
    model_signature: core.Sig2D,
    rho: float,
    trunc: int = 4,
    rk_subdivs: int = 50,
    integral_subdivs: int = 100,
    r_bs: float = None,
    vol_bs: float = None,
    cache = None
) -> float:
    """
    Fair price of a European call option under a signature volatility model.
    """
    # Fix the cache initialization logic
    if cache is None or cache is False:
        cache = core.compute_shuffle_cache(trunc)

    k_0 = np.log(initial_price / strike)
    variance_reduction = (r_bs is not None) and (vol_bs is not None)
    
    # Precompute the squared signature on the C++ side if needed
    model_sig_squared = core.shuffle(model_signature, model_signature, trunc, cache)
    upper_bound = 500.0

    if variance_reduction:
        # High-speed C++ compiled VR integrand
        def integrand_vr(u):
            return core.european_call_integrand_vr(
                u, k_0, maturity, model_signature, model_sig_squared, 
                rho, r_bs, vol_bs, trunc, rk_subdivs, upper_bound, cache
            )
        
        integral, _ = scipy.integrate.quad(integrand_vr, 0, np.inf, limit=integral_subdivs, epsabs=0.1)
        bs_call = european_call_bs(initial_price, maturity, strike, r_bs, vol_bs)
        return bs_call - strike / np.pi * integral

    else:
        # High-speed C++ compiled non-VR integrand
        def integrand(u):
            return core.european_call_integrand(
                u, k_0, maturity, model_signature, model_sig_squared, 
                rho, trunc, rk_subdivs, upper_bound, cache
            )
        
        integral, _ = scipy.integrate.quad(integrand, 0, np.inf, limit=integral_subdivs, epsabs=0.1)
        return initial_price - strike / np.pi * integral