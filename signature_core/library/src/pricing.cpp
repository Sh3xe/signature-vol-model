#include "pricing.hpp"
#include "utilities.hpp"

using namespace std::complex_literals;

Sig2D psi_derivative(cdouble u, double t, const Sig2D &psi, const Sig2D &model_sig, const Sig2D &model_sig_squared, double rho, int trunc, const std::shared_ptr<ShuffleCache> &cache)
{
	auto psi_1 = projection_on(psi, 0b0, 1);
	auto psi_2 = projection_on(psi, 0b1, 1);
	
	auto res = shuffle(psi_2, psi_2, trunc, cache);
	res *= 0.5;

	auto b = shuffle(model_sig, psi_2, trunc, cache);
	b *= ( rho * 1i * u );
	res += b;

	auto d = model_sig_squared.copy();
	cdouble scalar = 0.5*(-(u*u) - 1i*u);
	d *= scalar;
	res += d;

	auto e = projection_on(psi, 0b11, 2);
	e *= 0.5;
	res += e;

	res += psi_1;

	return res;
}

Sig2D simulate_psi_rk4(const Sig2D &psi_0, cdouble u, double maturity, const Sig2D &model_sig, const Sig2D &model_sig_squared, int rk_subdivs, double rho, int trunc, const std::shared_ptr<ShuffleCache> &cache)
{
	auto psi = psi_0.copy();
	double dt = maturity / static_cast<double>(rk_subdivs);
	for( size_t i = 0; i < rk_subdivs; ++i )
	{
		double di = static_cast<double>(i);

		auto k1 = psi_derivative(u, di*dt, psi, model_sig, model_sig_squared, rho, trunc, cache);
		auto k2 = psi_derivative(u, di*dt+dt/2, psi + 0.5*dt*k1, model_sig, model_sig_squared, rho, trunc, cache);
		auto k3 = psi_derivative(u, di*dt+dt/2, psi + 0.5*dt*k2, model_sig, model_sig_squared, rho, trunc, cache);
		auto k4 = psi_derivative(u, (di+1)*dt, psi + dt*k3, model_sig, model_sig_squared, rho, trunc, cache);

		psi += (k1 + 2*k2 + 2*k3 + k4)*(dt/6);
	}

	return psi;
}

/**
 * @brief Returns f(u) = e^{i(u-i/2)k_0 + psi_0} where psi follows [...]
 * 
 * @param u integrand parameter
 * @param k_0 log(S_K)
 * @param maturity option parameter
 * @param model_sig option parameter
 * @param model_sig_squares MUST be shuffle(model_sig, model_sig, trunc) (pre computed to gain time)
 * @param rho signature volatility parameter (correlation of the global & the model brownians)
 * @param r_bs Black-Scholes risk-free rate (used for variance reduction)
 * @param vol_bs Black-Scholes volatility (used for variance reduction)
 * @param trunc signature truncation
 * @param rk_subdivs subdivision used for the Runge-Kutta 4 method used to solve the Ricatti PDE
 * @param upper_bound if u > upper_bound, returns 0.0; (used for numerical stability)
 * @param cache ShuffleProduct Cache (must be present)
 * @return double f(u)
 */
double european_call_integrand_vr(
	double u,
    double k_0, double maturity,
    const Sig2D &model_sig, const Sig2D &model_sig_squared, double rho,
	double r_bs, double vol_bs,
    int trunc, int rk_subdivs,
	double upper_bound,
	std::shared_ptr<ShuffleCache> cache )
{
	if( u > upper_bound) return 0.0;

	auto psi0 = Sig2D(trunc, 0.0);

	cdouble us = static_cast<cdouble>(u)-0.5i;

	Sig2D res = simulate_psi_rk4(psi0, us, maturity, model_sig, model_sig_squared, rk_subdivs, rho, trunc, cache);

	cdouble characteristic_bs = std::exp(-0.5*vol_bs*vol_bs *(us*us+1i*us) * maturity);

	cdouble characteristic_val = std::exp( res.get_element(0b0, 0) );

	cdouble t = ( std::exp(1i*us*k_0) * (characteristic_val - characteristic_bs) ) / ( u*u + 0.25 );

	return t.real();
}