#pragma once

#include <complex>
#include <string>
#include <ostream>

using cdouble = std::complex<double>;
class Signature;

std::ostream &operator<<(std::ostream &os, const Signature &sig);

std::string to_string(const Signature &sig);

inline bool equal(const cdouble a, const cdouble b, const float eps = 1e-5)
{
    const cdouble diff = a - b;
    return ( fabs(diff.real())+fabs(diff.real()) ) < eps;
}