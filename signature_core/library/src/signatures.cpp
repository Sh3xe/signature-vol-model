#include "signatures.hpp"
#include <array>
#include <iostream>
#include <bitset>

Signature::Signature(size_t order, cdouble fill_value):
    m_order(order)
{
    // [order] tensors of increasing size: 2^0 + ... + 2^{order} = 2^{order+1}-1
    size_t n_elements = (1ULL << (order+1)) - 1;

    // allocated the data and fill it
    m_data.resize(n_elements, fill_value);
}

cdouble Signature::get_element( const coords &coordinates ) const
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

Signature &Signature::operator-=(const Signature &other)
{
    for(size_t i = 0; i < std::min(m_data.size(), other.m_data.size()); ++i)
        m_data[i] -= other.m_data[i];
    return *this;
}

Signature &Signature::operator/=(cdouble constant)
{
    for(size_t i = 0; i < m_data.size(); ++i)
        m_data[i] /= constant;
    return *this;
}

Signature &Signature::operator*=(cdouble constant)
{
    for(size_t i = 0; i < m_data.size(); ++i)
        m_data[i] *= constant;
    return *this;
}

// Signature projection_on( const Signature &sig, const coords &coordinates )
// {
//     Signature res( std::max( static_cast<size_t>(0), sig.m_data.size() - coordinates.size() ), 0.0);

//     if( coordinates.size() == 0 )
//         return res;
    
//     // assert self.dim > max(indices), f"Incompatible dimension: the dimension of u is at least {max(indices)} but the signature has an inner dimension of {self.dim}"

//     // res = [ np.zeros( (self.dim,)*i, dtype=self.dtype ) for i in range(len(self.data)-len(indices)) ]

    

// //     for i in range(len(self.data)-len(indices)):
// //         for index in np.ndindex( (self.dim,) * i ):
// //             tensor_prod_index = tuple(index) + indices
// //             res[i][index] = self.data[len(tensor_prod_index)][tensor_prod_index]

//     return res;
// }

template <typename ShuffleBasisAccessor>
Signature shuffle_base(
    const Signature &left, const Signature &right, size_t truncation,
    ShuffleBasisAccessor &&shuffle_product_basis_acc)
{
    Signature out(truncation, 0.0);

    // Iterate over [left]'s tensors
    for(size_t il = 0; il <= std::min(left.order(), truncation); ++il)
    {
        size_t offset_l = (1ULL << il)-1;

        // Iterate over [right]'s tensors
        for(size_t ir = 0; ir <= std::min(right.order(), truncation-il) ; ++ir)
        {
            size_t offset_r = (1ULL << ir)-1;
            size_t offset_out = (1ULL << (ir+il))-1;
            
            // Iterate over all pairs of basis
            for(size_t jl = 0; jl < offset_l+1; ++jl)
            {
                cdouble left_val = left.m_data[offset_l+jl];
                if (left_val == 0.0) continue;

                for(size_t jr = 0; jr < offset_r+1; ++jr)
                {
                    cdouble right_val = right.m_data[offset_r+jr];
                    if (right_val == 0.0) continue;

                    // Shuffle product is linear, "shuffle_product_basis" computes this product on a basis,
                    // we call it on all possible basis, and multiply it be their value
                    // the last few parameters of "shuffle_product_basis" (out, constant, begin_index) allows us
                    // to directly write into a signature without allocating any memory
                    shuffle_product_basis_acc(
                        jl, il,
                        jr, ir,
                        out.m_data, offset_out,
                        left_val*right_val
                    );
                }
            }
        }
    }

    return out;
}

Signature shuffle(const Signature &left, const Signature &right, size_t truncation, const std::shared_ptr<ShuffleCache> &cache)
{
    auto accessor = [&](
        uint32_t jl, uint32_t il,
        uint32_t jr, uint32_t ir,
        std::vector<cdouble> &output,
        uint32_t offset_out,
        cdouble constant ) {
        auto &res = cache->get( jl, il, jr, ir);
        for(size_t i = 0; i < offset_out+1; ++i)
            output[offset_out+i] += res[i]*constant;
    };

    return shuffle_base(left, right, truncation, accessor);
}

Signature shuffle(const Signature &left, const Signature &right, size_t truncation)
{
    auto accessor = []( uint32_t jl, uint32_t il,
                        uint32_t jr, uint32_t ir,
                        std::vector<cdouble> &output,
                        uint32_t offset_out,
                        cdouble constant ) {
        shuffle_product_basis(
            jl, il,
            jr, ir,
            output, offset_out,
            constant
        );
    };

    return shuffle_base(left, right, truncation, accessor);
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

void shuffle_product_basis_rec(
    uint32_t left, uint32_t order_left, uint32_t right, uint32_t order_right,
    uint32_t curr_out, uint32_t cur_out_order,
    std::vector<cdouble> &out, uint32_t begin_index,
    cdouble constant )
{
    // write to the output at the proper index
    if(order_left == 0)
    {
        uint32_t index = (right << cur_out_order) + curr_out;
        out[begin_index+index] += constant;
        return;
    }

    // write to the output at the proper index
    if(order_right == 0)
    {
        uint32_t index = (left << cur_out_order) + curr_out;
        out[begin_index+index] += constant;
        return;
    }

    // Break [left]
    shuffle_product_basis_rec(
        left >> 1, order_left-1, right, order_right,
        ((left % 2) << cur_out_order) + curr_out, cur_out_order+1,
        out, begin_index, constant );
    
    // expand right
    shuffle_product_basis_rec(
        left, order_left, right >> 1, order_right-1,
        ((right % 2) << cur_out_order) + curr_out, cur_out_order+1,
        out, begin_index, constant );
}

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
 * @param begin_index instead of writing the result in out[...], will write it to out[begin_index + ...]
 **/
void shuffle_product_basis(
    uint32_t left, uint32_t order_left,
    uint32_t right, uint32_t order_right,
    std::vector<cdouble> &out, uint32_t begin_index,
    cdouble constant )
{
    shuffle_product_basis_rec(left, order_left, right, order_right, 0b0, 0, out, begin_index, constant);
}

std::shared_ptr<ShuffleCache> compute_shuffle_cache(uint32_t truncation)
{
    ShuffleCache *cache = new ShuffleCache;
    cache->truncation = truncation;
    cache->memory_usage = 0;

    for(uint32_t order_i = 0; order_i <= truncation; ++order_i)
    for(uint32_t order_j = 0; order_j <= truncation-order_i; ++order_j)
    {
        for(uint32_t i = 0; i < (1 << order_i); ++i )
        for(uint32_t j = 0; j < (1 << order_j); ++j )
        {
            std::vector<cdouble> output ( (1<<(order_i+order_j)) , 0.0);

            shuffle_product_basis(i, order_i, j, order_j, output, 0, 1.0);

            uint64_t key = ShuffleCache::make_key(i, order_i, j, order_j);
            cache->memory_usage += output.size() * sizeof(cdouble);
            cache->cache[key] = std::move(output);
        }
    }

    return std::shared_ptr<ShuffleCache>(cache);
}

cdouble bracket(const Signature &left, const Signature &right)
{
    cdouble total = 0.0;

    for(size_t i = 0; i < std::min(left.m_data.size(), right.m_data.size()); ++i)
        total += left.m_data[i]*right.m_data[i];

    return total;
}

Signature projection_on( const Signature &sig, uint32_t coordinates, uint32_t coord_order )
{
    Signature out ( std::max( static_cast<size_t>(0), sig.order() - coord_order), 0.0);

    size_t cur_order = 0;
    for(size_t i = 0; i < out.m_data.size(); ++i)
    {
        // which tensor are we updating?
        // i = 0 -> order 0; i=1.2 -> order 1; i=3..6 -> order 1; i=2^n-1...2^(n+1)-2 -> order n
        bool need_order_update = i == ((1 << (cur_order+1)) - 1);
        if( need_order_update ) ++cur_order;

        // We "concatenate" the two coords: so 0b11001 and 0b001 will be 0b[11001][001]
        size_t shifted_i = (coordinates << cur_order) + i - (1<<cur_order) + 1;
        out.m_data[i] = sig.get_element(shifted_i, cur_order+coord_order);
    }

    return out;
}