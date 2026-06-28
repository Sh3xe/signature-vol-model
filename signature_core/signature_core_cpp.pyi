# signature_core/signature_core_cpp.pyi
from typing import List, Optional, Tuple, Union, overload
import numpy as np

class Sig2D:
    @overload
    def __init__(self, order: int, array: np.ndarray) -> None: ...
    @overload
    def __init__(self, order: int = ..., fill_value: complex = ...) -> None: ...

    def order(self) -> int: ...
    def copy(self) -> Sig2D: ...
    def get_element(self, coordinates: int, coord_order: int) -> complex: ...
    def get_element(self, coordinates: List[int]) -> complex: ...
    def set_element(self, coordinates: int, coord_order: int, el: complex) -> None: ...
    def set_element(self, coordinates: List[int], el: complex) -> None: ...

    # In-place operators
    def __iadd__(self, other: 'Sig2D') -> 'Sig2D': ...
    def __imul__(self, scalar: complex) -> 'Sig2D': ...
    def __itruediv__(self, scalar: complex) -> 'Sig2D': ...
    def __isub__(self, other: 'Sig2D') -> 'Sig2D': ...

    # Binary operators
    def __add__(self, other: 'Sig2D') -> 'Sig2D': ...
    def __sub__(self, other: 'Sig2D') -> 'Sig2D': ...
    def __mul__(self, scalar: complex) -> 'Sig2D': ...
    def __truediv__(self, scalar: complex) -> 'Sig2D': ...
    def __rmul__(self, scalar: complex) -> 'Sig2D': ...

    def __str__(self) -> str: ...

class ShuffleCache: ...

def compute_shuffle_cache(truncation: int) -> ShuffleCache: ...
def shuffle(left: Sig2D, right: Sig2D, truncation: int, cache: Optional[ShuffleCache] = None) -> Sig2D: ...
def matmul(left: Sig2D, right: Sig2D, truncation: int) -> Sig2D: ...
def bracket(left: Sig2D, right: Sig2D) -> complex: ...
def projection_on(sig: Sig2D, coordinates: int, coord_order: int) -> Sig2D: ...
def to_string(sig: Sig2D) -> str: ...

def european_call_integrand_vr(
    u: float,
    k_0: float,
    maturity: float,
    model_sig: Sig2D,
    model_sig_squared: Sig2D,
    rho: float,
    r_bs: float,
    vol_bs: float,
    trunc: int,
    rk_subdivs: int,
    upper_bound: float,
    cache: ShuffleCache
) -> float:
    """
    Returns f(u) = e^{i(u-i/2)k_0 + psi_0} using variance reduction for numerical stability.

    Args:
        u: Integrand parameter.
        k_0: log(S_0 / K) log-forward moneyness.
        maturity: Time to maturity of the option.
        model_sig: Model signature parameter.
        model_sig_squared: Precomputed shuffle(model_sig, model_sig) to optimize computation time.
        rho: Correlation parameter (global and model Brownian motions).
        r_bs: Black-Scholes risk-free rate used for variance reduction.
        vol_bs: Black-Scholes volatility used for variance reduction.
        trunc: Signature truncation order.
        rk_subdivs: Number of subdivisions for the Runge-Kutta 4 method.
        upper_bound: Numerical stability threshold (returns 0.0 if u > upper_bound).
        cache: Precomputed ShuffleCache instance.

    Returns:
        The real part of the integrand value.
    """
    ...

def european_call_integrand(
    u: float,
    k_0: float,
    maturity: float,
    model_sig: Sig2D,
    model_sig_squared: Sig2D,
    rho: float,
    trunc: int,
    rk_subdivs: int,
    upper_bound: float,
    cache: ShuffleCache
) -> float:
    """
    Returns f(u) = e^{i(u-i/2)k_0 + psi_0} without variance reduction.

    Args:
        u: Integrand parameter.
        k_0: log(S_0 / K) log-forward moneyness.
        maturity: Time to maturity of the option.
        model_sig: Model signature parameter.
        model_sig_squared: Precomputed shuffle(model_sig, model_sig) to optimize computation time.
        rho: Correlation parameter (global and model Brownian motions).
        trunc: Signature truncation order.
        rk_subdivs: Number of subdivisions for the Runge-Kutta 4 method.
        upper_bound: Numerical stability threshold (returns 0.0 if u > upper_bound).
        cache: Precomputed ShuffleCache instance.

    Returns:
        The real part of the integrand value.
    """
    ...

def european_call_sig(
    initial_price: float,
    maturity: float,
    strike: float,
    model_signature: Sig2D,
    rho: float,
    trunc: int,
    rk_subdivs: int,
    integral_subdivs: int,
    cache: ShuffleCache
) -> float:
    """
    Computes the fair price of a European call option under a signature volatility model 
    without using variance reduction.

    Args:
        initial_price: Initial asset price (S_0).
        maturity: Time to maturity of the option (T).
        strike: Option strike price (K).
        model_signature: Current state of the model path signature.
        rho: Correlation parameter between the asset price and volatility model driver.
        trunc: Truncation order for the signature vectors.
        rk_subdivs: Number of steps used in the RK4 scheme for solving the Riccati PDE.
        integral_subdivs: Maximum number of workspace subintervals allocated for the GSL QAGIU integrator.
        cache: A preallocated ShuffleCache instance used for accelerating tensor shuffle products.

    Returns:
        The fair value price of the European call option.
    """
    ...

def european_call_sig_vr(
    initial_price: float,
    maturity: float,
    strike: float,
    model_signature: Sig2D,
    rho: float,
    trunc: int,
    rk_subdivs: int,
    integral_subdivs: int,
    r_bs: float,
    vol_bs: float,
    cache: ShuffleCache
) -> float:
    """
    Computes the fair price of a European call option under a signature volatility model 
    using a Black-Scholes control variate technique for variance reduction.

    Args:
        initial_price: Initial asset price (S_0).
        maturity: Time to maturity of the option (T).
        strike: Option strike price (K).
        model_signature: Current state of the model path signature.
        rho: Correlation parameter between the asset price and volatility model driver.
        trunc: Truncation order for the signature vectors.
        rk_subdivs: Number of steps used in the RK4 scheme for solving the Riccati PDE.
        integral_subdivs: Maximum number of workspace subintervals allocated for the GSL QAGIU integrator.
        r_bs: Proxy Black-Scholes risk-free interest rate used to evaluate the control variate.
        vol_bs: Proxy Black-Scholes constant volatility parameter used to evaluate the control variate.
        cache: A preallocated ShuffleCache instance used for accelerating tensor shuffle products.

    Returns:
        The variance-reduced fair value price of the European call option.
    """
    ...