#pragma once

#include <optional>
#include <cmath>

#include "signatures.hpp"

double european_call_integrand_vr(
	double u,
    double k_0, double maturity,
    const Sig2D &model_sig, const Sig2D &model_sig_squared, double rho,
	double r_bs, double vol_bs,
    int trunc, int rk_subdivs,
	double upper_bound,
	std::shared_ptr<ShuffleCache> cache
);

double european_call_integrand(
    double u,
    double k_0, double maturity,
    const Sig2D &model_sig, const Sig2D &model_sig_squared, double rho,
    int trunc, int rk_subdivs,
    double upper_bound,
    std::shared_ptr<ShuffleCache> cache
);

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
);

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
);