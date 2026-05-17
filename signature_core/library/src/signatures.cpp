#include "signatures.hpp"

Signature::Signature(size_t order, cdouble fill_value):
    m_order(order)
{
    // [order] tensors of increasing size: 2^0 + ... + 2^{order} = 2^{order}-1
    size_t n_elements = (1ULL << order) - 1;

    // allocated the data and fill it
    m_data.resize(n_elements, fill_value);
}

size_t Signature::order()
{
    return m_order;
}

cdouble Signature::get_element( const coords &coordinates )
{
    return 0.0;
}

Signature &Signature::operator+=(const Signature &other)
{
    return *this;
}

Signature &Signature::operator*=(cdouble constant)
{
    return *this;
}

Signature &Signature::matmul(Signature &other)
{
    return *this;
}

Signature &Signature::shuffle(Signature &other)
{
    return *this;
}

Signature &Signature::projection_on( const coords &coordinates )
{
    return *this;
}