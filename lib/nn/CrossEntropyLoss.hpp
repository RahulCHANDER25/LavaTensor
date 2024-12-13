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
        const auto &inputData = input.tensor().datas();

        // Find max for numerical stability
        T maxVal = inputData[0];
        for (size_t i = 1; i < inputData.size(); ++i) {
            maxVal = std::max(maxVal, inputData[i]);
        }

        // Compute softmax and cross entropy loss
        T sum = 0;
        std::vector<T> ce(inputData.size());
        for (size_t i = 0; i < inputData.size(); ++i) {
            ce[i] = std::exp(inputData[i] - maxVal);
            sum += ce[i];
        }

        // Normalize and compute loss
        Tensor<T> output({1}, false);
        for (size_t i = 0; i < ce.size(); ++i) {
            ce[i] /= sum;
            if (i == targetIndex) {
                output[0] = -std::log(std::max(ce[i], epsilon));
            }
        }

        auto gradNode = std::make_shared<CrossEntropyLossBackward<T>>(input, targetIndex);
        output.setGradNode(gradNode);

        return output;
    }
};

} // namespace lava::nn
