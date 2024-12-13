/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** CrossEntropyLoss
*/

#pragma once

#include <cmath>
#include "../Tensor/Tensor.hpp"
#include "Module.hpp"

namespace lava::nn {

template <typename T>
class CrossEntropyLoss : public Module<T> {
    public:
    CrossEntropyLoss() : _lastInput({1}), _lastTarget({1}) {}

    Tensor<T> forward(Tensor<T> &input) override
    {
        (void)input;
        throw std::runtime_error("CrossEntropyLoss requires a target index. Use forward(input, targetIndex) instead.");
    }

    // Our specialized forward method for loss computation
    T forward(Tensor<T> &input, size_t targetIndex)
    {
        _lastInput = input;

        _lastTarget = Tensor<T>({static_cast<int>(input.tensor().datas().size())});
        auto &targetData = _lastTarget.tensor().datas();
        std::fill(targetData.begin(), targetData.end(), 0);
        targetData[targetIndex] = 1;

        const T epsilon = 1e-7;
        const auto &inputData = input.tensor().datas();

        // Find max for numerical stability
        T maxVal = inputData[0];
        for (size_t i = 1; i < inputData.size(); ++i) {
            maxVal = std::max(maxVal, inputData[i]);
        }

        // Compute softmax and cross entropy loss
        T sum = 0;
        std::vector<T> softmax(inputData.size());
        for (size_t i = 0; i < inputData.size(); ++i) {
            softmax[i] = std::exp(inputData[i] - maxVal);
            sum += softmax[i];
        }

        // Normalize and compute loss
        T loss = 0;
        for (size_t i = 0; i < softmax.size(); ++i) {
            softmax[i] /= sum;
            if (i == targetIndex) {
                loss = -std::log(std::max(softmax[i], epsilon));
            }
        }

        return loss;
    }

    private:
    Tensor<T> _lastInput;
    Tensor<T> _lastTarget;
};

} // namespace lava::nn
