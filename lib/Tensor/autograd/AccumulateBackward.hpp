/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** AccumulateBackward
*/

#pragma once

#include <iostream>
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
        //std::cout << "Accumulate: ";
        grad.dispRaw();
        _tensor.grad() += grad;
        _tensor.grad().dispRaw();
    }

    void backward() override
    {
        _tensor.grad() += 1;
    }

    virtual void auth() override { std::cout << "I am Accumulate\n"; }

private:
    Tensor<T> &_tensor;
};

}
