#include "signatures.hpp"
#include <iostream>
#include <chrono>
#include <bitset>

using namespace std::complex_literals;

bool check_accessors()
{
    // Tensor product test
    auto sig1 = Sig2D(3, 0.0);
    sig1.set_element({0, 0}, 1.2);
    sig1.set_element({0, 1}, -6.0);
    sig1.set_element({1, 0}, 0.08);
    sig1.set_element({1, 1}, 16.54);

    auto sig2 = Sig2D(3, 0.0);
    sig2.set_element({0}, 0.98);
    sig2.set_element({1}, -0.5);

    if( sig2.get_element({0}) != 0.98 || sig2.get_element({1}) != -0.5) {
        std::cout << "element accessors are broken" << std::endl;
        std::cout << sig2.get_element({0}) << " "  << sig2.get_element({1}) << std::endl;

        return false;
    }

    if( 
        sig1.get_element({0, 0}) != 1.2 || sig1.get_element({0, 1}) != -6.0 ||
        sig1.get_element({1, 0}) != 0.08 || sig1.get_element({1, 1}) != 16.54
    ) {
        std::cout << "element accessors are broken" << std::endl;
        std::cout << sig1.get_element({0, 0}) << " " << sig1.get_element({0, 1}) << std::endl;
        std::cout << sig1.get_element({1, 0}) << " " << sig1.get_element({1, 1}) << std::endl;

        return false;
    }

    if( sig2.get_element({0}) != sig2.get_element(0b0, 1) || sig2.get_element({1}) != sig2.get_element(0b1, 1)) {
        std::cout << "binary & vector api are not consistent with each other" << std::endl;
        std::cout << sig2.get_element({0}) << " "  << sig2.get_element({1}) << std::endl;
        std::cout << "VERSUS" << std::endl;
        std::cout << sig2.get_element(0b0, 1) << " "  << sig2.get_element(0b1, 1) << std::endl;
        return false;
    }

    if( 
        sig1.get_element({0, 0}) != sig1.get_element(0b00, 2) || sig1.get_element({0, 1}) != sig1.get_element(0b01, 2) ||
        sig1.get_element({1, 0}) != sig1.get_element(0b10, 2) || sig1.get_element({1, 1}) != sig1.get_element(0b11, 2)
    ) {
        std::cout << "binary & vector api are not consistent with each other" << std::endl;
        std::cout << sig1.get_element({0, 0}) << " " << sig1.get_element({0, 1}) << std::endl;
        std::cout << sig1.get_element({1, 0}) << " " << sig1.get_element({1, 1}) << std::endl;
        std::cout << "VERSUS" << std::endl;
        std::cout << sig1.get_element(0b00, 2) << " " << sig1.get_element(0b01, 2) << std::endl;
        std::cout << sig1.get_element(0b10, 2) << " " << sig1.get_element(0b11, 2) << std::endl;
        return false;
    }

    return true;
}

bool check_matmul()
{
    // Tensor product test
    auto sig1 = Sig2D(2, 0.0);
    sig1.set_element({0}, 0.0);
    sig1.set_element({1}, 1.0);

    auto sig2 = Sig2D(2, 0.0);
    sig2.set_element({0}, 1.0);
    sig2.set_element({1}, 0.0);

    auto res = matmul(sig1, sig2, 3);

    if( 
        res.get_element({0, 0}) != 0.0 || res.get_element({0, 1}) != 0.0 ||
        res.get_element({1, 0}) != 1.0 || res.get_element({1, 1}) != 0.0
    ) {
        std::cout << "matmul is broken" << std::endl;
        std::cout << res << std::endl;

        return false;
    }

    sig1.set_element({0, 0}, 1.2);
    sig1.set_element({0, 1}, -6.0);
    sig1.set_element({1, 0}, 0.08);
    sig1.set_element({1, 1}, 16.54);

    sig2.set_element({0}, 0.98);
    sig2.set_element({1}, -5.0);

    res = matmul(sig1, sig2, 3);

    double target_hash = -47.5164;
    double hash = 0.0;
    for(size_t i1 = 0; i1 < 2; ++i1)
    for(size_t i2 = 0; i2 < 2; ++i2)
    for(size_t i3 = 0; i3 < 2; ++i3) {
        hash += res.get_element({i1,i2,i3}).real();
    }

    if( std::abs(hash-target_hash) > 0.01)
    {
        std::cout << "matmul is broken" << std::endl;
        std::cout << res << std::endl;
        return false;
    }

    return true;
}

void print_cache_perf()
{
    auto time_begin = std::chrono::system_clock::now();

    // output for the shuffle product 12 [shuffle] 12
    uint32_t max_order = 10;
    std::vector<cdouble> output ( (1<<max_order) , 0.0);
    for(uint32_t order_i = 0; order_i < max_order; ++order_i)
    for(uint32_t order_j = 0; order_j < max_order-order_i; ++order_j)
    {
        for(uint32_t i = 0; i < (1 << order_i); ++i )
        for(uint32_t j = 0; j < (1 << order_j); ++j )
        {
            shuffle_product_basis(i, order_i, j, order_j, output, 0, 1.0);
        }
    }

    auto time_end = std::chrono::system_clock::now();

    std::cout << "Time to compute all possibilities: "
        // output the number of ms
        << std::chrono::duration<float, std::ratio<1,1000>>(time_end - time_begin).count() 
        << "ms" << std::endl;

    time_begin = std::chrono::system_clock::now();

    auto cache = compute_shuffle_cache(10);

    time_end = std::chrono::system_clock::now();

    std::cout << "Time to compute the cache: "
        // output the number of ms
        << std::chrono::duration<float, std::ratio<1,1000>>(time_end - time_begin).count() 
        << "ms" << std::endl;
    std::cout << "Cache size : " << cache->memory_usage << std::endl;

    time_begin = std::chrono::system_clock::now();

    auto &vec = cache->get(0b01, 2, 0b01, 2);
    for(uint32_t order_i = 0; order_i < max_order; ++order_i)
    for(uint32_t order_j = 0; order_j < max_order-order_i; ++order_j)
    {
        for(uint32_t i = 0; i < (1 << order_i); ++i )
        for(uint32_t j = 0; j < (1 << order_j); ++j )
        {
            cache->get(i, order_i, j, order_j);
        }
    }

    time_end = std::chrono::system_clock::now();
    std::cout << "Time to compute all possibilities using the cache: "
        // output the number of ms
        << std::chrono::duration<float, std::ratio<1,1000>>(time_end - time_begin).count() 
        << "ms" << std::endl;
}

bool check_shuffle_product()
{
    Sig2D left(1, 0.0);
    left.set_element(0b0, 0, 1.24);
    left.set_element(0b0, 1, -10.0 );
    left.set_element(0b1, 1, 1.0 );
    
    Sig2D right(2, 0.0);
    right.set_element(0b00, 2, -2.5);
    right.set_element(0b11, 2, -2.5);

    // Compute via the baseline shuffle implementation
    Sig2D res1 = shuffle(left, right, left.order() + right.order());

    // Compute via the cached shuffle implementation
    auto cache = compute_shuffle_cache(10);
    Sig2D res2 = shuffle(left, right, left.order() + right.order(), cache);

    // 2. Validate Order 0, 1, and 2 element-by-element for both results
    // Expected Order 0: [ (0,0) ]
    // Expected Order 1: [ (0,0), (0,0) ]
    // Expected Order 2: [[ (-3.1,0), (0,0) ], [ (0,0), (-3.1,0) ]]
    auto check_low_orders = [](Sig2D& res) -> bool {
        if (res.get_element(0b0, 0).real() != 0.0 ||
            res.get_element(0b0, 1).real() != 0.0 ||
            res.get_element(0b1, 1).real() != 0.0 ||
            std::abs(res.get_element(0b00, 2).real() - (-3.1)) > 1e-6 ||
            res.get_element(0b01, 2).real() != 0.0 ||
            res.get_element(0b10, 2).real() != 0.0 ||
            std::abs(res.get_element(0b11, 2).real() - (-3.1)) > 1e-6) 
        {
            return false;
        }
        return true;
    };

    if (!check_low_orders(res1)) {
        std::cout << "Baseline shuffle: low-order components are broken" << std::endl;
        std::cout << res1 << std::endl;
        return false;
    }

    if (!check_low_orders(res2)) {
        std::cout << "Cached shuffle: low-order components are broken" << std::endl;
        std::cout << res2 << std::endl;
        return false;
    }

    return true;
}

bool check_projection()
{
    Sig2D left(1, 0.0);
    left.set_element(0b0, 0, 1.24);
    left.set_element(0b0, 1, -10.0 );
    left.set_element(0b1, 1, 1.0 );
    
    Sig2D right(2, 0.0);
    right.set_element(0b00, 2, -2.5);
    right.set_element(0b11, 2, -2.5);

    Sig2D res = shuffle(left, right, left.order() + right.order());

    Sig2D projected_res = projection_on(res, 0b1, 1);

    if( !(
        projected_res.get_element(0b0, 0) == res.get_element(0b1, 1) &&

        projected_res.get_element(0b0, 1) == res.get_element(0b10, 2) &&
        projected_res.get_element(0b1, 1) == res.get_element(0b11, 2) &&

        projected_res.get_element(0b00, 2) == res.get_element(0b100, 3) &&
        projected_res.get_element(0b01, 2) == res.get_element(0b101, 3) &&
        projected_res.get_element(0b10, 2) == res.get_element(0b110, 3) &&
        projected_res.get_element(0b11, 2) == res.get_element(0b111, 3)
    ))
    {
        std::cout << "Project does not work : " << projected_res << std::endl;
        return false;
    }

    return true;
}

int main()
{
    if( check_accessors() )
    {
        std::cout << "Element accessors OK" << std::endl;
    }

    if( check_matmul() )
    {
        std::cout << "Matmul OK" << std::endl;
    }

    if( check_shuffle_product() )
    {
        std::cout << "Shuffle product OK" << std::endl;
    }

    print_cache_perf();

    if( check_projection() )
    {
        std::cout << "Projection OK" << std::endl;
    }

    return 1;
}