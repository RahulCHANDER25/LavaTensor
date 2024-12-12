/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ReLU
*/

#pragma once

#include <algorithm>
#include "../Tensor/Tensor.hpp"
#include "Module.hpp"

namespace lava::nn {

template <typename T>
class ReLU : public Module<T> {
    public:
    ReLU() : _lastInput({1}) {}

    Tensor<T> forward(const Tensor<T> &input) override
    {
        _lastInput = input;
        const auto &inputData = input.tensor().datas();
        Tensor<T> output({static_cast<int>(inputData.size())});
        auto &outputData = output.tensor().datas();

        // ReLU forward: max(0, x)
        for (size_t i = 0; i < inputData.size(); ++i) {
            outputData[i] = std::max(static_cast<T>(0), inputData[i]);
        }
        return output;
    }

    Tensor<T> &backward(Tensor<T> &gradOutput) override
    {
        const auto &inputData = _lastInput.tensor().datas();
        auto &gradData = gradOutput.tensor().datas();
        // ReLU backward: gradient is 1 if input > 0, 0 otherwise
        for (size_t i = 0; i < gradData.size(); ++i) {
            gradData[i] = inputData[i] > 0 ? gradData[i] : 0;
        }
        return gradOutput;
    }

    private:
    Tensor<T> _lastInput;
};

} // namespace lava::nn
