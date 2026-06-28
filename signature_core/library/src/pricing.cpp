#include "pricing.hpp"
#include "utilities.hpp"

#include <cmath>
#include <iostream>
#include <algorithm>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

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

	cdouble res_0 = res.get_element(0b0, 0);
	if (std::isnan(res_0.real()) || std::isnan(res_0.imag())  || std::isinf(res_0.real())) return 0.0;

	cdouble characteristic_bs = std::exp(-0.5*vol_bs*vol_bs *(us*us+1i*us) * maturity);
	
	cdouble characteristic_val = std::exp( res_0 );

	cdouble t = ( std::exp(1i*us*k_0) * (characteristic_val - characteristic_bs) ) / ( u*u + 0.25 );

    if( t.real() < 0.0 || t.real() > 1000.0 ) return 0.0;

	return t.real();
}

double european_call_integrand(
    double u,
    double k_0, double maturity,
    const Sig2D &model_sig, const Sig2D &model_sig_squared, double rho,
    int trunc, int rk_subdivs,
    double upper_bound,
    std::shared_ptr<ShuffleCache> cache )
{
    auto psi0 = Sig2D(trunc, 0.0);

    cdouble us = static_cast<cdouble>(u) - 0.5i;

    // Solve the Riccati PDE via RK4
    Sig2D res = simulate_psi_rk4(psi0, us, maturity, model_sig, model_sig_squared, rk_subdivs, rho, trunc, cache);

	cdouble res_0 = res.get_element(0b0, 0);
	if (std::isnan(res_0.real()) || std::isnan(res_0.imag())  || std::isinf(res_0.real())) return 0.0;

    // Extract the characteristic function value (res.data[0] equivalent)
    cdouble characteristic_val = std::exp( res_0 );

    // Compute the integrand without subtracting the Black-Scholes component
    cdouble t = ( std::exp(1i * us * k_0) * characteristic_val ) / ( u * u + 0.25 );

    if( t.real() < 0.0 || t.real() > 1000.0 ) return 0.0;

    return t.real();
}

double european_call_bs(double initial_price, double maturity, double strike, double r_bs, double vol_bs)
{
    if (vol_bs <= 0.0 || maturity <= 0.0) return std::max(0.0, initial_price - strike);

    double d1 = (std::log(initial_price / strike) + (r_bs + 0.5 * vol_bs * vol_bs) * maturity) / (vol_bs * std::sqrt(maturity));

    double d2 = d1 - vol_bs * std::sqrt(maturity);

    return initial_price * 0.5 * std::erfc(-d1 / std::sqrt(2.0)) - strike * std::exp(-r_bs * maturity) * 0.5 * std::erfc(-d2 / std::sqrt(2.0));
}

double european_call_bs_vega(double risk_free_rate, double maturity, double K, double S, double volatility)
{
    if (volatility <= 0.0 || maturity <= 0.0) return 0.0;

    double d1 = (std::log(S / K) + (risk_free_rate + 0.5 * volatility * volatility) * maturity) / 
                (volatility * std::sqrt(maturity));

    double norm_pdf = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * d1 * d1);

    return S * std::sqrt(maturity) * norm_pdf;
}

std::optional<double> newton_iv(
    double time_to_maturity, 
    double risk_free_rate, 
    double strike, 
    double price, 
    double option_price)
{
    // 1. Fundamental arbitrage checks before spinning up the loop
    double intrinsic_value = std::max(0.0, price - strike * std::exp(-risk_free_rate * time_to_maturity));
    if (option_price < intrinsic_value || option_price >= price) {
        return std::nullopt; 
    }

    double x0 = 0.2; // Initial volatility guess (20%)
    constexpr double tolerance = 1e-7;
    constexpr int maxiter = 50;

    for(int i = 0; i < maxiter; ++i)
    {
        double f = european_call_bs(price, time_to_maturity, strike, risk_free_rate, x0) - option_price;
        double v = european_call_bs_vega(risk_free_rate, time_to_maturity, strike, price, x0);
        
        // Guard against division by zero or extremely flat Vega regions
        if (std::abs(v) < 1e-12) {
            break;
        }
        
        double x1 = x0 - f / v;
        
        if (x1 <= 0.0) {
            x1 = 0.0001;
        }

        if (std::abs(x1 - x0) <= tolerance * std::abs(x1)) {
            return x1;
        }
        
        x0 = x1;
    }
    
    return std::nullopt;
}

struct VrIntegrandContext
{
    double k_0; double maturity; const Sig2D &model_sig; const Sig2D &model_sig_squared;
    double rho; double r_bs; double vol_bs; int trunc; int rk_subdivs; double upper_bound;
    std::shared_ptr<ShuffleCache> cache;
};

struct NonVrIntegrandContext
{
    double k_0; double maturity; const Sig2D &model_sig; const Sig2D &model_sig_squared;
    double rho; int trunc; int rk_subdivs; double upper_bound;
    std::shared_ptr<ShuffleCache> cache;
};

static double gsl_integrand_vr_wrapper(double u, void *params)
{
    auto *ctx = static_cast<VrIntegrandContext*>(params);
    return european_call_integrand_vr(
        u, ctx->k_0, ctx->maturity, ctx->model_sig, ctx->model_sig_squared,
        ctx->rho, ctx->r_bs, ctx->vol_bs, ctx->trunc, ctx->rk_subdivs, ctx->upper_bound, ctx->cache
    );
}

static double gsl_integrand_non_vr_wrapper(double u, void *params)
{
    auto *ctx = static_cast<NonVrIntegrandContext*>(params);
    return european_call_integrand(
        u, ctx->k_0, ctx->maturity, ctx->model_sig, ctx->model_sig_squared,
        ctx->rho, ctx->trunc, ctx->rk_subdivs, ctx->upper_bound, ctx->cache
    );
}

double european_call_sig(
    double initial_price,
    double maturity,
    double strike,
    const Sig2D &model_signature,
    double rho,
    int trunc,
    int rk_subdivs,
    int integral_subdivs,
    std::shared_ptr<ShuffleCache> cache
) {
    double k_0 = std::log(initial_price / strike);
    gsl_set_error_handler_off();
    
    // Precompute squared signature
    Sig2D model_sig_squared = shuffle(model_signature, model_signature, trunc, cache);
    double upper_bound = 500.0;

    // Allocate adaptive quadrature workspace
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(integral_subdivs);
    double integral_result = 0.0;
    double abs_error = 0.0;

    gsl_function F;
	NonVrIntegrandContext ctx{
		k_0, maturity, model_signature, model_sig_squared,
		rho, trunc, rk_subdivs, upper_bound, cache
	};
	F.function = &gsl_integrand_non_vr_wrapper;
	F.params = &ctx;

	gsl_integration_qagiu(&F, 0.0, 0.1, 1e-6, integral_subdivs, w, &integral_result, &abs_error);
	gsl_integration_workspace_free(w);

	return initial_price - (strike / M_PI) * integral_result;
}

double european_call_sig_vr(
    double initial_price,
    double maturity,
    double strike,
    const Sig2D &model_signature,
    double rho,
    int trunc,
    int rk_subdivs,
    int integral_subdivs,
    double r_bs,
    double vol_bs,
    std::shared_ptr<ShuffleCache> cache
) {
    double k_0 = std::log(initial_price / strike);
    gsl_set_error_handler_off();
    
    // Precompute squared signature
    Sig2D model_sig_squared = shuffle(model_signature, model_signature, trunc, cache);
    double upper_bound = 500.0;

    // Allocate adaptive quadrature workspace
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(integral_subdivs);
    double integral_result = 0.0;
    double abs_error = 0.0;

    gsl_function F;
	VrIntegrandContext ctx{
		k_0, maturity, model_signature, model_sig_squared,
		rho, r_bs, vol_bs, trunc, rk_subdivs, upper_bound, cache
	};
	F.function = &gsl_integrand_vr_wrapper;
	F.params = &ctx;

	// Adaptive integration mapping [0, inf)
	gsl_integration_qagiu(&F, 0.0, 0.1, 1e-7, integral_subdivs, w, &integral_result, &abs_error);
	gsl_integration_workspace_free(w);

	double bs_call = european_call_bs(initial_price, maturity, strike, r_bs, vol_bs);
	return bs_call - (strike / M_PI) * integral_result;
}