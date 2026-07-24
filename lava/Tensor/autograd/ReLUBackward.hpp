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
        if (_reluRes.device()->isCPU()) {
            for (size_t i = 0; i < _reluRes.datas().size(); i++) {
                _reluRes[i] = _reluRes[i] > T{0}; // if x > 0, grad = 1 else grad = 0
            }
        } else {
            std::vector<T> hostData(_reluRes.storage()->size());
            _reluRes.device()->copyDeviceToHost(hostData.data(), _reluRes.storage()->data(), hostData.size());
            for (size_t i = 0; i < hostData.size(); i++) {
                hostData[i] = (hostData[i] > T{0}) ? T{1} : T{0};
            }
            _reluRes.device()->copyHostToDevice(_reluRes.storage()->data(), hostData.data(), hostData.size());
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
