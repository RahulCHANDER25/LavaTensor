/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** AddBackward
*/

#pragma once

#include <iostream>
#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/GradNode.hpp"

namespace lava {

template <typename T>
class AddBackward : public GradNode<T> {
    public:
    AddBackward(Tensor<T> &tensorA, Tensor<T> &tensorB) : lava::GradNode<T>(), _onesArr(tensorA.tensor())
    {
        if (_onesArr.device()->isCPU()) {
            std::fill(_onesArr.datas().begin(), _onesArr.datas().end(), T{1});
        } else {
            std::vector<T> temp(_onesArr.storage()->size(), T{1});
            _onesArr.device()->copyHostToDevice(_onesArr.storage()->data(), temp.data(), temp.size());
        }

        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(tensorB.gradNode());
    }

    AddBackward(Tensor<T> &tensorA) : lava::GradNode<T>(), _onesArr(tensorA.tensor())
    {
        if (_onesArr.device()->isCPU()) {
            std::fill(_onesArr.datas().begin(), _onesArr.datas().end(), T{1});
        } else {
            std::vector<T> temp(_onesArr.storage()->size(), T{1});
            _onesArr.device()->copyHostToDevice(_onesArr.storage()->data(), temp.data(), temp.size());
        }

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
            std::cout << "Ones array !\n";
            std::cout << _onesArr.shape()[1] << std::endl;
            this->_nextGrads[0]->backward(_onesArr);
        }
        if (this->_nextGrads[1]) {
            std::cout << "Ones array !\n";
            std::cout << _onesArr.shape()[1] << std::endl;
            this->_nextGrads[1]->backward(_onesArr);
        }
    }

    private:
    TensorArray<T> _onesArr;
};

} // namespace lava
