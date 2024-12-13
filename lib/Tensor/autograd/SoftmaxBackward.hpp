/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ReLUBackward
*/

#pragma once

#include <stdexcept>
#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class SoftmaxBackward : public GradNode<T> {
    public:
    SoftmaxBackward(Tensor<T> &input) : _res(input.tensor()) {}

    ~SoftmaxBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        throw std::runtime_error("Not implemented");
    }

    void backward() override
    {
        throw std::runtime_error("Not implemented");
    }

    private:
    TensorArray<T> _res;
};
} // namespace lava
