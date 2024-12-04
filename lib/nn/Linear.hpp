/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Linear
*/

#pragma once

#include "nn/Module.hpp"

namespace lava::nn {

template <typename T>
class Linear : public Module<T> { // Careful maybe again template specification on types
public:
    Linear(int inFeatures, int outFeatures):
    _weights({outFeatures, inFeatures}),
    _biases({outFeatures, 1})
    {}

    ~Linear() override = default;

    Tensor<T> forward(const Tensor<T> &) override
    {
        // x * _weights.tensor().transpose() + _biases; // Do this again
        // x * W (T) + B // => Implement Matrix Vector matmul
        // Add 1 dimension at the end for that (for the vector).
        // All is done in autograd
        // Just this and it is okkk
        return {2, 3};
    }

private:
    Tensor<T> _weights;
    Tensor<T> _biases;
};

}
