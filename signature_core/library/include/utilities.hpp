#pragma once

#include <complex>

using cdouble = std::complex<double>;

inline bool equal(const cdouble a, const cdouble b, const float eps = 1e-5)
{
    const cdouble diff = a - b;
    return ( fabs(diff.real())+fabs(diff.real()) ) < eps;
}