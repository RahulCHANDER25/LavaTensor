/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Module
*/

#pragma once

#include "../Tensor/Tensor.hpp"

namespace lava::nn {

template <typename T>
class Module {
    public:
    virtual ~Module() = default;

    virtual Tensor<T> forward(const Tensor<T> &input) = 0;

    virtual Tensor<T> &backward(Tensor<T> &gradOutput)
    {
        return gradOutput; // Default implementation
    }

    Tensor<T> operator()(const Tensor<T> &input)
    {
        return forward(input);
    }
};

} // namespace lava::nn
