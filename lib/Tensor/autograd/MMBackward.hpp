/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** MMBackward
*/

#pragma once

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
        _tensorBCpy(tensorB.tensor()),
        _newGradA(tensorA.tensor().shape(), tensorA.tensor().strides()),
        _newGradB(tensorB.tensor().shape(), tensorB.tensor().strides())
    {
        this->_nextGrads.push_back(tensorA.gradNode());
        this->_nextGrads.push_back(tensorB.gradNode());
    }

    ~MMBackward() override = default;

    void backward(TensorArray<T> grad) override
    {
        for (int i = 0; i < _tensorACpy.shape()[0]; i++) {
            for (int j = 0; j < _tensorACpy.shape()[1]; j++) {
                for (int k = 0; k < _tensorBCpy.shape()[1]; k++) {
                    _newGradA({i, j}) += _tensorBCpy({j, k});
                    _newGradB({j, k}) += _tensorACpy({i, j});
                }
            }
        }

        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(grad * _newGradA);
        }
        if (this->_nextGrads[1]) {
            this->_nextGrads[1]->backward(grad * _newGradB);
        }
    }

    void backward() override
    {
        for (int i = 0; i < _tensorACpy.shape()[0]; i++) {
            for (int j = 0; j < _tensorACpy.shape()[1]; j++) {
                for (int k = 0; k < _tensorBCpy.shape()[1]; k++) {
                    _newGradA({i, j}) += _tensorBCpy({j, k});
                    _newGradB({i, j}) += _tensorACpy({i, j});
                }
            }
        }
        if (this->_nextGrads[0]) {
            this->_nextGrads[0]->backward(_newGradA);
        }
        if (this->_nextGrads[1]) {
            this->_nextGrads[1]->backward(_newGradB);
        }
    }

private:
    TensorArray<T> _tensorACpy;
    TensorArray<T> _tensorBCpy;

    TensorArray<T> _newGradA;
    TensorArray<T> _newGradB;
};

}
