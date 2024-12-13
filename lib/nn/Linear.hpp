/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Linear
*/

#pragma once

#include <cmath>
#include "Module.hpp"
#include "Tensor/Tensor.hpp"

namespace lava::nn {

template <typename T>
class Linear : public Module<T> {
public:
    Linear(int inFeatures, int outFeatures):
        _weights({inFeatures, outFeatures}, true),
        _biases({outFeatures}, true)
    {
    }

    ~Linear() override = default;

    Tensor<T> forward(Tensor<T> &x) override
    {
        return x.matmul(this->_weights) + _biases;
    }

    // Only weights and biases as Tensor
    Tensor<T> _weights;
    Tensor<T> _biases;
private:
};

}
