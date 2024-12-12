/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** MMBackward
*/

#pragma once

#include <cstdio>
#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class MMBackward : public GradNode<T> {
public:
    MMBackward(Tensor<T> &tensorA, Tensor<T> &tensorB):
        lava::GradNode<T>(),
        _tensorACpy(tensorA.tensor()),
        _tensorBCpy(tensorB.tensor())
    {
        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(tensorB.gradNode());
    }

    ~MMBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        // For A: grad_A = grad_C × B^T
        if (this->_nextGrads[0]) {
            auto bTransposed = _tensorBCpy;
            bTransposed.transposed();
            this->_nextGrads[0]->backward(grad.matmul(bTransposed));
        }

        // For B: grad_B = A^T × grad_C
        if (this->_nextGrads[1]) {
            auto aTransposed = _tensorACpy;
            aTransposed.transposed();
            this->_nextGrads[1]->backward(aTransposed.matmul(grad));
        }
    }

    void backward() override
    {
        // This should never be called without a gradient
        TensorArray<T> ones(_tensorACpy.shape(), _tensorACpy.strides());
        std::fill(ones.datas().begin(), ones.datas().end(), 1);
        backward(ones);
    }

private:
    TensorArray<T> _tensorACpy;
    TensorArray<T> _tensorBCpy;
};

}
