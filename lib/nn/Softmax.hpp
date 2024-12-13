/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Softmax
*/

#pragma once

#include <cmath>
#include "../Tensor/Tensor.hpp"
#include "Module.hpp"

namespace lava::nn {

template <typename T>
class Softmax : public Module<T> {
    public:
    Softmax() : _lastOutput({1}) {}

    // Implement softmax the call it inside CrossEntropy !
    // Tensor<T> softmax(Tensor<T> &input)
    // {
    //     for (size_t i = 0; i < input.datas().size(); i++) {

    //     }
    // }

    Tensor<T> forward(Tensor<T> &input) override
    {
        const auto &inputData = input.tensor().datas();
        _lastOutput = Tensor<T>({static_cast<int>(inputData.size())});

        // Find max for numerical stability
        T maxVal = inputData[0];
        for (size_t i = 1; i < inputData.size(); ++i) {
            maxVal = std::max(maxVal, inputData[i]);
        }

        // Compute exp(x - max) and sum
        T sum = 0;
        auto &outputData = _lastOutput.tensor().datas();
        for (size_t i = 0; i < inputData.size(); ++i) {
            outputData[i] = std::exp(inputData[i] - maxVal);
            sum += outputData[i];
        }

        // Normalize
        for (size_t i = 0; i < outputData.size(); ++i) {
            outputData[i] /= sum;
        }

        return _lastOutput;
    }

    private:
    Tensor<T> _lastOutput;
};

} // namespace lava::nn
