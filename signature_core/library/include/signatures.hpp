#pragma once

#include <cstdint>
#include <vector>
#include <complex>

using cdouble = std::complex<double>;
using coords = std::vector<uint32_t>;

// Assumed dimension: 2
class Signature
{
public:
    Signature(size_t order, cdouble fill_value = 0.0);

    size_t order();
    cdouble get_element( const coords &coordinates );

    Signature &operator+=(const Signature &other);
    Signature &operator*=(cdouble constant);

    Signature &matmul(Signature &other);
    Signature &shuffle(Signature &other);

    Signature &projection_on( const coords &coordinates );

private:
    uint32_t m_order;
    std::vector<cdouble> m_data;
};

// def signature( x: np.ndarray, trunc: int ):
// def bracket(sig_left, sig_right):
// def bracket_with_process(cst_sig, process_sig, max_order = None):