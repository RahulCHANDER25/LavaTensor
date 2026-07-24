/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** CrossEntropyLoss
*/

#pragma once

#include <cmath>
#include "Module.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/autograd/CrossEntropyLossBackward.hpp"

namespace lava::nn {

template <typename T>
class CrossEntropyLoss {
    public:
    CrossEntropyLoss() = default;

    // Our specialized forward method for loss computation
    Tensor<T> forward(Tensor<T> &input, size_t targetIndex)
    {
        const T epsilon = 1e-7;

        std::vector<T> hostInput;
        if (input.tensor().device()->isCPU()) {
            hostInput = input.tensor().datas();
        } else {
            hostInput.resize(input.tensor().storage()->size());
            input.tensor().device()->copyDeviceToHost(hostInput.data(), input.tensor().storage()->data(), hostInput.size());
        }

        // Find max for numerical stability
        T maxVal = hostInput[0];
        for (size_t i = 1; i < hostInput.size(); ++i) {
            maxVal = std::max(maxVal, hostInput[i]);
        }

        // Compute softmax and cross entropy loss
        T sum = 0;
        std::vector<T> ce(hostInput.size());
        for (size_t i = 0; i < hostInput.size(); ++i) {
            ce[i] = std::exp(hostInput[i] - maxVal);
            sum += ce[i];
        }

        // Normalize and compute loss
        T lossVal = 0;
        for (size_t i = 0; i < ce.size(); ++i) {
            ce[i] /= sum;
            if (i == targetIndex) {
                lossVal = -std::log(std::max(ce[i], epsilon));
            }
        }

        Tensor<T> output({1}, input.tensor().device());
        if (output.tensor().device()->isCPU()) {
            output[0] = lossVal;
        } else {
            output.tensor().device()->copyHostToDevice(output.tensor().storage()->data(), &lossVal, 1);
        }

        auto gradNode = std::make_shared<CrossEntropyLossBackward<T>>(input, targetIndex);
        output.setGradNode(gradNode);

        return output;
    }
};

} // namespace lava::nn
