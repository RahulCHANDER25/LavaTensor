/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** TensorArray
*/

#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <vector>

#include "TensorBase.hpp"
#include <initializer_list>

namespace lava {

/**
 * @tparam Type of the underlying datas of the Tensor.
 *
 * @brief TensorArray class with all the TensorBase basic operations.
 * 
 * NOTE: This class only does Tensors computations needed, no the gradient Computations
 */
template <typename T> 
class TensorArray : public TensorBase {
public:
    TensorArray(std::initializer_list<int> shape);

    TensorArray(const TensorArray &tensor);

    ~TensorArray() override  = default;

    void dispRaw() override;

    TensorArray operator+(TensorArray &oth) { return _tensorApplyOperator(oth, std::plus<T>()); }
    TensorArray operator-(TensorArray &oth) { return _tensorApplyOperator(oth, std::minus<T>()); };
    TensorArray operator*(TensorArray &oth) { return _tensorApplyOperator(oth, std::multiplies<T>()); };
    TensorArray operator/(TensorArray &oth)
    {
        return _tensorApplyOperator(
            oth,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    };

    TensorArray operator+(T k) { return _scalarApplyOperator(k, std::plus<T>()); }
    TensorArray operator-(T k) { return _scalarApplyOperator(k, std::minus<T>()); };
    TensorArray operator*(T k) { return _scalarApplyOperator(k, std::multiplies<T>()); };
    TensorArray operator/(T k)
    {
        return _scalarApplyOperator(
            k,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    };

    T operator()(std::initializer_list<int>) const; // TODO: Changing to variadic arguments
    T& operator()(std::initializer_list<int> indexes);

    T operator[](size_t idx) const;
    T &operator[](size_t idx);

private:

    // Checks of shape to be done !
    TensorArray _tensorApplyOperator(TensorArray &oth, std::function<T (const T &, const T &)> func);

    TensorArray _scalarApplyOperator(T k, std::function<T (const T &, const T &)> func);

    static size_t getStride(int k, const std::vector<int> &shape);

    std::vector<int> _shape; /** Shape of the Tensor */
    std::vector<int> _strides; /** Stride of the Tensor */

    std::vector<T> _datas; /** Underlying datas of the Tensor */
};

}

/**
 * Supported types of TensorArray
 */

template class lava::TensorArray<int>;
template class lava::TensorArray<size_t>;
template class lava::TensorArray<double>;
template class lava::TensorArray<float>;
