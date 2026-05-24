#pragma once

#include "utilities.hpp"

#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>

struct ShuffleCache;

using coords = std::vector<size_t>;

// Assumed dimension: 2
class Signature
{
public:
    Signature(size_t order, cdouble fill_value = 0.0);

    constexpr size_t order() const { return m_order; }

    cdouble get_element( uint32_t coordinates, uint32_t coord_order ) const
    {
        size_t shift = (1<<coord_order)-1 + coordinates;

        if( shift > m_data.size() ) return 0.0;

        return m_data[shift];
    }

    void set_element( uint32_t coordinates, uint32_t coord_order, cdouble el )
    {
        size_t shift = (1<<coord_order)-1 + coordinates;

        if( shift > m_data.size() ) return;

        m_data[shift] = el;
    }

    cdouble get_element( const coords &coordinates ) const;

    void set_element( const coords &coordinates, cdouble el );

    Signature &operator+=(const Signature &other);

    Signature &operator-=(const Signature &other);

    Signature &operator*=(cdouble constant);

    Signature &operator/=(cdouble constant);

public:
    size_t m_order;
    std::vector<cdouble> m_data;
};

struct ShuffleCache
{
    // The maximum tensor order used
    size_t truncation;

    // memory used in bytes
    size_t memory_usage;

    // maps 4 numbers (left, order_left, right, order_right) into a corresponding vector of size 2^(order_left+order_right)
    // that corresponds to the result of shuffle_product_basis(left, order_left, right, order_right, ...)
    std::unordered_map< uint64_t, std::vector<cdouble> > cache;

    // Bit-packing utility: Packs 4 numbers into a single 64-bit word
    // Assumes orders fit in 16 bits, indices fit in 16 bits.
    static inline uint64_t make_key(
        uint32_t left, uint32_t order_left, 
        uint32_t right, uint32_t order_right) noexcept 
    {
        return (static_cast<uint64_t>(left)       << 48) |
               (static_cast<uint64_t>(order_left) << 32) |
               (static_cast<uint64_t>(right)      << 16) |
               (static_cast<uint64_t>(order_right));
    }

    /**
     * Same as shuffle_product_basis(left, order_left, right_order_right, ...) but fetched the aleady cached result.
     */
    inline const std::vector<cdouble>& get(
        uint32_t left, uint32_t order_left, 
        uint32_t right, uint32_t order_right) const 
    {
        uint64_t key = make_key(left, order_left, right, order_right);
        auto it = cache.find(key);
        
        if (it == cache.end()) [[unlikely]] {
            throw std::out_of_range("Requested basis combination not found in ShuffleCache.");
        }
        
        return it->second; // Returns reference directly from the map bucket
    }
};

std::shared_ptr<ShuffleCache> compute_shuffle_cache(uint32_t truncation);

Signature shuffle(const Signature &left, const Signature &right, size_t truncation);

Signature shuffle(const Signature &left, const Signature &right, size_t truncation, const std::shared_ptr<ShuffleCache> &cache);

Signature matmul(const Signature &left, const Signature &right, size_t truncation);

cdouble bracket(const Signature &left, const Signature &right);

Signature projection_on( const Signature &sig, uint32_t coordinates, uint32_t coord_order );

// def signature( x: np.ndarray, trunc: int ):
// def bracket_with_process(cst_sig, process_sig, max_order = None):

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
    cdouble constant );