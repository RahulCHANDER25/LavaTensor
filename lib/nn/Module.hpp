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

    virtual Tensor<T> forward(Tensor<T> &/* input */) = 0;

    // Eval mode function ==> Put all submodules to eval and tensors too
};

}
