/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** AddBackward
*/

#pragma once

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class AddBackward : public GradNode<T> {
public:
    AddBackward(Tensor<T> &tensorA, Tensor<T> &tensorB):
        lava::GradNode<T>(),
        _onesArr(tensorA.tensor())
    {
        std::fill(_onesArr.datas().begin(), _onesArr.datas().end(), T{1});

        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(tensorB.gradNode());
    }

    AddBackward(Tensor<T> &tensorA):
        lava::GradNode<T>(),
        _onesArr(tensorA.tensor())
    {
        std::fill(_onesArr.datas().begin(), _onesArr.datas().end(), T{1});

        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(nullptr);
    }

    ~AddBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(grad * _onesArr);
        }
        if (this->_nextGrads[1]) {
            this->_nextGrads[1]->backward(grad * _onesArr);
        }
    }

    void backward() override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_onesArr);
        }
        if (this->_nextGrads[1]) {
            this->_nextGrads[1]->backward(_onesArr);
        }
    }

private:
    TensorArray<T> _onesArr;
};

}
