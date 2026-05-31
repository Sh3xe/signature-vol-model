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