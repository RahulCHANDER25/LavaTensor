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

#include <initializer_list>

namespace lava {

/**
 *  @tparam Type of the underlying datas of the Tensor.
 *
 *  @brief TensorArray class with all the TensorBase basic operations.
 * 
 *  NOTE: This class only does Tensors computations needed, no the gradient Computations
 */
template <typename T> 
class TensorArray {
public:
    enum class InitType {
        ZERO,
        RANDOM,
        RANGE
    };

    /**
     *  @brief Default constructor is deleted because a shape is need when intializing a Tensor.
    */
    TensorArray() = delete;

    /**
     *  @brief Constructor of TensorArray with the initial shape of the tensor.
     *
     *  @param shape Initial shape of the tensor created
     *
     *  NOTE: This constructor inits the strides with the shape and
     *       it inits the underlying datas randomly.
     */
    TensorArray(std::initializer_list<int> shape);

    /**
     *  @brief Constructor of TensorArray with the initial shape of the tensor.
     *
     *  @param shape Initial shape of the tensor created
     *  @param type Initialization type of the elements in the newly created tensor
     *
     *  NOTE: This constructor inits the strides with the shape and
     *       it inits the underlying datas randomly.
     */
    TensorArray(std::initializer_list<int> shape, InitType type);

    /**
     *  @brief Constructor of TensorArray a shape and a strides given as parameters as vectors.
     *
     *  @param shape Shape given to the new Tensor created
     *  @param strides Strides given to the new Tensor created
     *
     *  NOTE: This constructor inits the strides with the shape and
     *        it inits the underlying datas with default value of @tparam T (eg. `0` for `int`).
     */
    TensorArray(const std::vector<int> &shape, const std::vector<int> &strides);

    /**
     *  @brief Copy constructor of the TensorArray class
     *
     *  @param tensor Constant reference to a Tensor array that will be copied.
     *
     *  NOTE: This constructor copy everything, that includes the strides and the shape.
     */
    TensorArray(const TensorArray &tensor);

    /**
     *  @brief Copy constructor of the TensorArray class
     *
     *  @param datas Constant reference to a vector with the underlying datas of the tensor.
     *
     *  NOTE: The shape is the size of the @param datas vector, and the strides in 1.
     */
    TensorArray(const std::vector<T> &datas);

    /**
     *  @brief Default destructor of the TensorArray class
     */
    ~TensorArray()  = default;

    /**
     *  @brief Display the underlying tensor's data in raw format
     *         without taking account the shape or the strides
     */
    void dispRaw();

    /**
     *  @brief Perform a matrix multiplication between `this` and @param oth.
     *
     *  @param oth The other Tensor with which we will perform matrix multiplication
     *  @return A new Tensor result of matrix multiplication
     *
     *  NOTE: Only 2D Tensors are currently supported for matrix multiplication
     */
    TensorArray matmul(const TensorArray &oth);

    TensorArray &operator=(const TensorArray<T> &oth);
    TensorArray &operator=(TensorArray<T> &&oth) noexcept;

    TensorArray operator+(const TensorArray &oth) { return _tensorOperation(oth, std::plus<T>()); }
    TensorArray operator-(const TensorArray &oth) { return _tensorOperation(oth, std::minus<T>()); }
    TensorArray operator*(const TensorArray &oth) { return _tensorOperation(oth, std::multiplies<T>()); }
    TensorArray operator/(const TensorArray &oth)
    {
        return _tensorOperation(
            oth,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    }

    TensorArray &operator+=(TensorArray &oth) { return _inPlaceTensorOperation(oth, std::plus<T>()); }
    TensorArray &operator-=(TensorArray &oth) { return _inPlaceTensorOperation(oth, std::minus<T>()); }
    TensorArray &operator*=(TensorArray &oth) { return _inPlaceTensorOperation(oth, std::multiplies<T>()); }
    TensorArray &operator/=(TensorArray &oth)
    {
        return _inPlaceTensorOperation(
            oth,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    }

    TensorArray operator+(T k) { return _scalarOperation(k, std::plus<T>()); }
    TensorArray operator-(T k) { return _scalarOperation(k, std::minus<T>()); }
    TensorArray operator*(T k) { return _scalarOperation(k, std::multiplies<T>()); }
    TensorArray operator/(T k)
    {
        return _scalarOperation(
            k,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    }

    TensorArray &operator+=(T k) { return _inPlaceScalarOperation(k, std::plus<T>()); }
    TensorArray &operator-=(T k) { return _inPlaceScalarOperation(k, std::minus<T>()); }
    TensorArray &operator*=(T k) { return _inPlaceScalarOperation(k, std::multiplies<T>()); }
    TensorArray &operator/=(T k)
    {
        return _inPlaceScalarOperation(
            k,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    }

    T operator()(std::initializer_list<int> indexes) const; // TODO: Changing to variadic arguments
    T& operator()(std::initializer_list<int> indexes);

    T operator[](size_t idx) const;
    T &operator[](size_t idx);

    /**
     *  @brief Returns the tensor's shape as a vector
     *
     *  @return Reference to the Tensor's shape as a vector
     */
    std::vector<int> &shape() { return _shape; }

    /**
     *  @brief Returns the tensor's shape as a vector
     *
     *  @return Const Reference to the Tensor's shape as a vector
     */
    const std::vector<int> &shape() const { return _shape; }

    /**
     *  @brief Returns the tensor' strides as a vector
     *
     *  @return Tensor' strides as a vector
     */
    std::vector<int> &strides() { return _strides; }

    /**
     *  @brief Returns the tensor' strides as a vector
     *
     *  @return Tensor' strides as a vector
     */
    const std::vector<int> &strides() const { return _strides; }

    /**
     *  @brief Returns the tensor's underlying datas
     *
     *  @return Tensor's underlying datas as a vector
     */
    std::vector<T> &datas() { return _datas; }

private:

    // TODO: Checks of shape to be done !

    /**
     *  @brief Do an in-place operation specified by @param func that takes another tensor @param oth on the current tensor.
     *         This operation takes the elements of `this` and @param oth one by one and perform the operation.
     *
     *  @param oth The other tensor that will be used to modify the current one.
     *  @param func Operation take 2 values and return a single result that modify the current Tensor.
     *
     *  @return Reference to the current Tensor that has the result of the operations.
     */
    TensorArray &_inPlaceTensorOperation(const TensorArray &oth, std::function<T (const T &, const T &)> func);

    /**
     *  @brief Do an in-place operation specified by @param func that takes another tensor @param oth on the current tensor.
     *         This operation takes the elements of `this` and @param oth one by one and perform the operation
     *         to create a new Tensor
     *
     *  @param oth The other tensor that will be used to compute the new one
     *  @param func Operation take 2 values and return a single result that is added to the Tensor returned.
     *
     *  @return A new tensor which has the result of the operation done.
     */
    TensorArray _tensorOperation(const TensorArray &oth, std::function<T (const T &, const T &)> func);

    /**
     *  @brief Do an in-place operation specified, by @param func , that takes a scalar @param oth , on the current tensor.
     *         This operation takes the elements of `this` one by one and perform an in-place operation with k.
     *
     *  @param k Scalar that will be used by @param func for the binary operation done on the current tensor.
     *  @param func Operation take 2 values and return a single result that modify the current Tensor.
     *
     *  @return Reference to the current Tensor that has the result of the operations.
     */
    TensorArray &_inPlaceScalarOperation(T k, std::function<T (const T &, const T &)> func);

    /**
     *  @brief Do an in-place operation specified, by @param func , that takes a scalar @param oth , on the current tensor.
     *         This operation takes the elements of `this` one by one and perform an in-place operation with k.
     *
     *  @param k Scalar that will be used by @param func for the binary operation done on the new tensor.
     *  @param func Operation take 2 values and return a single result that modify the new Tensor.
     *
     *  @return A new Tensor that has the result of the operations.
     */
    TensorArray _scalarOperation(T k, std::function<T (const T &, const T &)> func);

    static size_t getStride(int k, const std::vector<int> &shape);

    std::vector<int> _shape; /** Shape of the Tensor */
    std::vector<int> _strides; /** Stride of the Tensor */

    std::vector<T> _datas; /** Underlying datas of the Tensor */
};

}

/**
 * Supported types of TensorArray class
 */

template class lava::TensorArray<int>;
template class lava::TensorArray<size_t>;
template class lava::TensorArray<double>;
template class lava::TensorArray<float>;

// Iterators for strides based operations ? ==> Duro
// Utils directory

// Documentation
