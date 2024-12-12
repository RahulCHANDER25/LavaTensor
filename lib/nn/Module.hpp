/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Module
*/

#pragma once

#include "Tensor/Tensor.hpp"

namespace lava::nn {

template <typename T>
class Module {
    public:
    virtual ~Module() = default;

    virtual Tensor<T> forward(Tensor<T> &input) = 0;

    Tensor<T> operator()(const Tensor<T> &input)
    {
        return forward(input);
    }
};

} // namespace lava::nn
