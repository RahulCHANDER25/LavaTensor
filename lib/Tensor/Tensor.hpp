/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Tensor
*/

#pragma once

#include <memory>
#include "Tensor/TensorArray.hpp"

namespace lava {

template <typename T>
class Tensor;

template <typename T>
using BackwardFunction = std::function<void(const Tensor<T> &)>;

template <typename T>
class Tensor {
public:

    Tensor() = delete;

    Tensor(std::initializer_list<int> shape);
    Tensor(const Tensor &tensor);
    Tensor(const TensorArray<T> &tensorArray);
    Tensor(TensorArray<T> data, bool requiresGrad = false);

    void backward();
    void zeroGrad();
    bool requiresGrad() const { return _requiresGrad; }
    void setRequiresGrad(bool requiresGrad) { _requiresGrad = requiresGrad; }

    void dispRaw()
    {
        _tensor.dispRaw();
        // dprintf(1, "Hello I am a Tensor\n");
    }

    Tensor matmul(const Tensor &oth);

    Tensor &operator=(const Tensor<T> &oth);
    Tensor &operator=(Tensor<T> &&oth) noexcept;

    Tensor operator+(Tensor &oth);
    Tensor operator-(Tensor &oth);
    Tensor operator*(Tensor &oth);
    Tensor operator/(Tensor &oth);

    Tensor operator+(T k);
    Tensor operator-(T k);
    Tensor operator*(T k);
    Tensor operator/(T k);

    TensorArray<T> &tensor() { return _tensor; }
    TensorArray<T> &grad() { return _grad; }
    const TensorArray<T> &tensor() const { return _tensor; }
    const TensorArray<T> &grad() const { return _grad; }

    T operator()(std::initializer_list<int> indexes) const; // TODO: Changing to variadic arguments
    T& operator()(std::initializer_list<int> indexes);

    T operator[](size_t idx) const;
    T &operator[](size_t idx);

private:
    TensorArray<T> _tensor; /** TensorArray with the tensor datas */
    TensorArray<T> _grad; /** TensorArray with the tensor gradient => Maybe std::optionnal for non-leaf and inference mode ? */
    bool _requiresGrad{false};

    std::vector<std::shared_ptr<Tensor>> _previous;
    BackwardFunction<T> _gradFn;

    static Tensor createWithGrad(
        TensorArray<T> data,
        std::vector<std::shared_ptr<Tensor<T>>> prev,
        BackwardFunction<T> gradFn
    );

    // std::shared_ptr<BackwardFunc> _node;
    /** Node (BackwardFunc associated) of the current tensor for the backpropagation
        => No node if it is not a leaf, but the backwardFunc is still added to the graph
    */
};

}

/**
 * Supported types of Tensor class
 */

template class lava::Tensor<int>;
template class lava::Tensor<size_t>;
template class lava::Tensor<double>;
template class lava::Tensor<float>;
