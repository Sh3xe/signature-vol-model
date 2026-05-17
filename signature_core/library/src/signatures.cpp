#include "signatures.hpp"

Signature::Signature(size_t order, cdouble fill_value):
    m_order(order)
{
    // [order] tensors of increasing size: 2^0 + ... + 2^{order} = 2^{order+1}-1
    size_t n_elements = (1ULL << (order+1)) - 1;

    // allocated the data and fill it
    m_data.resize(n_elements, fill_value);
}

cdouble Signature::get_element( const coords &coordinates )
{
    size_t coor_size = std::min(coordinates.size(), m_order);
    size_t el_id = (1 << coor_size )-1;

    for(size_t i = 0; i < coor_size; ++i)
    {
        size_t power_of_two = (1ULL << (coor_size-1-i));
        el_id += power_of_two*coordinates[i];
    }

    return m_data[el_id];
}

void Signature::set_element( const coords &coordinates, cdouble el )
{
    size_t coor_size = std::min(coordinates.size(), m_order);
    size_t el_id = (1 << coor_size )-1;
    
    for(size_t i = 0; i < coor_size; ++i)
    {
        size_t power_of_two = (1ULL << (coor_size-1-i));
        el_id += power_of_two*coordinates[i];
    }

    m_data[el_id] = el;
}

Signature &Signature::operator+=(const Signature &other)
{
    for(size_t i = 0; i < std::min(m_data.size(), other.m_data.size()); ++i)
        m_data[i] += other.m_data[i];
    return *this;
}

Signature &Signature::operator*=(cdouble constant)
{
    for(size_t i = 0; i < m_data.size(); ++i)
        m_data[i] *= constant;
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

Signature matmul(const Signature &left, const Signature &right, size_t truncation)
{
    Signature out(truncation, 0.0);

    // Iterate over [left]'s tensors
    for(size_t il = 0; il <= left.order(); ++il)
    {
        size_t offset_l = (1ULL << il)-1;

        // Iterate over [right]'s tensors
        for(size_t ir = 0; ir <= right.order(); ++ir)
        {
            size_t offset_r = (1ULL << ir)-1;
            size_t offset_out = (1ULL << (ir+il))-1;
            
            // Perform the tensor product
            for(size_t jl = 0; jl < offset_l+1; ++jl)
            {
                for(size_t jr = 0; jr < offset_r+1; ++jr)
                {
                    // by convention, the last dimension is the contiguous one
                    size_t idx_out = offset_out+jr+(offset_r+1)*jl;
                    out.m_data[idx_out] += left.m_data[offset_l+jl]*right.m_data[offset_r+jr];
                }
            }
        }
    }

    return out;
}