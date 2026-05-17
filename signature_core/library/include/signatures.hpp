#pragma once

#include <cstdint>
#include <vector>
#include <complex>

using cdouble = std::complex<double>;
using coords = std::vector<size_t>;

// Assumed dimension: 2
class Signature
{
public:
    Signature(size_t order, cdouble fill_value = 0.0);

    constexpr size_t order() const { return m_order; }
    cdouble get_element( const coords &coordinates );
    void set_element( const coords &coordinates, cdouble el );

    Signature &operator+=(const Signature &other);
    Signature &operator*=(cdouble constant);

    Signature &shuffle(Signature &other);

    Signature &projection_on( const coords &coordinates );

    friend Signature matmul(const Signature &left, const Signature &right, size_t truncation);
    friend std::ostream &operator<<(std::ostream &os, const Signature &sig);

private:
    size_t m_order;
    std::vector<cdouble> m_data;
};

Signature matmul(const Signature &left, const Signature &right, size_t truncation);

// def signature( x: np.ndarray, trunc: int ):
// def bracket(sig_left, sig_right):
// def bracket_with_process(cst_sig, process_sig, max_order = None):