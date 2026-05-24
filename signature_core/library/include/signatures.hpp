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

/**
 * @brief Computes the Shuffle product between two basis tensors of dimension 2 and arbitrary order.
 *
 * Example: shuffle(12212, 1211) produces an output tensor of order 5+3=8.
 *
 * Basis tensors are represented as integers where each digit corresponds to a basis element.
 * For instance, the tensor 1221 is represented as left=0b0110 with order_left=4.
 *
 * @param left Integer representation of the left basis tensor.
 * @param order_left Order of the left basis tensor.
 * @param right Integer representation of the right basis tensor.
 * @param order_right Order of the right basis tensor.
 * @param[out] out Output vector. Must have at least 2^(order_left+order_right) elements.
 */
void shuffle_product_basis(uint32_t left, uint32_t order_left, uint32_t right, uint32_t order_right, std::vector<cdouble> &out);

// def signature( x: np.ndarray, trunc: int ):
// def bracket(sig_left, sig_right):
// def bracket_with_process(cst_sig, process_sig, max_order = None):