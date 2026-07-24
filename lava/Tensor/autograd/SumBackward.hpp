/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** SumBackward
*/

#pragma once

#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class SumBackward : public GradNode<T> {
public:
    SumBackward(Tensor<T> &tensor):
        lava::GradNode<T>(),
        _onesArr(tensor.tensor())
    {
        if (_onesArr.device()->isCPU()) {
            std::fill(_onesArr.datas().begin(), _onesArr.datas().end(), T{1});
        } else {
            std::vector<T> temp(_onesArr.storage()->size(), T{1});
            _onesArr.device()->copyHostToDevice(_onesArr.storage()->data(), temp.data(), temp.size());
        }

        this->_nextGrads.push_back(tensor.gradNode());
    }

    ~SumBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(grad * _onesArr);
        }
    }

    void backward() override
    {
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_onesArr);
        }
    }

private:
    TensorArray<T> _onesArr;
};

}
