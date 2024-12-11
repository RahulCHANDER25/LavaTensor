/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** MulBackward
*/

#pragma once

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class MulBackward : public GradNode<T> {
public:
    MulBackward(Tensor<T> &tensorA, Tensor<T> &tensorB):
        lava::GradNode<T>(),
        _tensorACpy(tensorA.tensor()),
        _tensorBCpy(tensorB.tensor())
    {
        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(tensorB.gradNode());
    }

    MulBackward(Tensor<T> &tensorA, T k):
        lava::GradNode<T>(),
        _tensorACpy(tensorA.tensor()),
        _tensorBCpy(tensorA.tensor().shape(), tensorA.tensor().strides())
    {
        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(nullptr);

        std::fill(_tensorBCpy.datas().begin(), _tensorBCpy.datas().end(), k);
    }

    ~MulBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(grad * _tensorBCpy);
        }
        if (this->_nextGrads[1]) {
            this->_nextGrads[1]->backward(grad * _tensorACpy);
        }
    }

    void backward() override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_tensorBCpy);
        }
        if (this->_nextGrads[1]) {
            this->_nextGrads[1]->backward(_tensorACpy);
        }
    }

private:
    TensorArray<T> _tensorACpy;
    TensorArray<T> _tensorBCpy;
};

}
