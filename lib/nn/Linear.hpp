/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Linear
*/

#pragma once

#include <iostream>
#include "nn/Module.hpp"

namespace lava::nn {

template <typename T>
class Linear : public Module<T> { // Careful maybe again template specification on types
public:
    Linear(int inFeatures, int outFeatures):
    _weights({inFeatures, outFeatures}, true),
    _biases({outFeatures, 1}, true)
    {}

    ~Linear() override = default;

    Tensor<T> forward(Tensor<T> &x) override
    {
        _weights.dispRaw();
        auto out = x.matmul(_weights); // + _biases; // Matrix Vector product
        _biases.dispRaw();
        return out;
    }

    Tensor<T> _weights;
    Tensor<T> _biases;
private:
};

}
