/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ReLUBackward
*/

#pragma once

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class ReLUBackward : public GradNode<T> {
public:
    ReLUBackward(Tensor<T> &input):
        _reluRes(input.tensor())
    {
        for (size_t i = 0; i < _reluRes.datas().size(); i++) {
            _reluRes[i] = _reluRes[i] > 0; // if x > 0, grad = 1 else grad = 0
        }
        this->_nextGrads.push_back(input.gradNode());
    }

    ~ReLUBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(grad * _reluRes);
        }
    }

    void backward() override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_reluRes);
        }
    }

private:
    TensorArray<T> _reluRes;
};

}
