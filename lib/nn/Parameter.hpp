#pragma once

#include "../Tensor/Tensor.hpp"
#include "../Tensor/TensorArray.hpp"

namespace lava::nn {

template <typename T>
class Parameter {
    public:
    Parameter() = default;

    Parameter(const std::vector<int> &shape, const std::vector<int> &strides)
        : _data(TensorArray<T>(shape, strides), true), _grad(TensorArray<T>(shape, strides), true)
    {
    }

    const Tensor<T> &tensor() const
    {
        return _data;
    }

    Tensor<T> &tensor()
    {
        return _data;
    }

    const Tensor<T> &grad() const
    {
        return _grad;
    }

    Tensor<T> &grad()
    {
        return _grad;
    }

    private:
    Tensor<T> _data;
    Tensor<T> _grad;
};

} // namespace lava::nn
