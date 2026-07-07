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
class CrossEntropyLossBackward : public GradNode<T> {
    public:
    CrossEntropyLossBackward(Tensor<T> &input, size_t targetIndex) : _res(input.tensor()), _targetIndex(targetIndex)
    {
        this->_nextGrads.push_back(input.gradNode());
    }

    ~CrossEntropyLossBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        _res[_targetIndex] -= 1;
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_res * grad);
        }
    }

    void backward() override
    {
        _res[_targetIndex] -= 1;
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_res);
        }
    }

    private:
    TensorArray<T> _res;
    size_t _targetIndex;
};
} // namespace lava
