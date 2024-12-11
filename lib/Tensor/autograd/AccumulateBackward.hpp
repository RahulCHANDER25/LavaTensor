/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** AccumulateBackward
*/

#pragma once

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class AccumulateBackward : public GradNode<T> {
public:
    AccumulateBackward(Tensor<T> &tensor):
        lava::GradNode<T>(),
        _tensor(tensor)
    {
    }

    ~AccumulateBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        _tensor.grad() += grad;
    }

    void backward() override
    {
        _tensor.grad() += 1;
    }

private:
    Tensor<T> &_tensor;
};

}
