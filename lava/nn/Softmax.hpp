/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Softmax
*/

#pragma once

#include <cmath>
#include "Module.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/autograd/SoftmaxBackward.hpp"

namespace lava::nn {

template <typename T>
class Softmax : public Module<T> {
    public:
    Softmax() = default;

    // Implement softmax the call it inside CrossEntropy !
    // Tensor<T> softmax(Tensor<T> &input)
    // {
    //     for (size_t i = 0; i < input.datas().size(); i++) {

    //     }
    // }

    Tensor<T> softmax(Tensor<T> &input)
    {
        const auto &inputData = input.tensor().datas();
        auto output = Tensor<T>({static_cast<int>(inputData.size())});

        // Find max for numerical stability
        T maxVal = inputData[0];
        for (size_t i = 1; i < inputData.size(); ++i) {
            maxVal = std::max(maxVal, inputData[i]);
        }

        // Compute exp(x - max) and sum
        T sum = 0;
        auto &outputData = output.tensor().datas();
        for (size_t i = 0; i < inputData.size(); ++i) {
            outputData[i] = std::exp(inputData[i] - maxVal);
            sum += outputData[i];
        }

        // Normalize
        for (size_t i = 0; i < outputData.size(); ++i) {
            outputData[i] /= sum;
        }

        return output;
    }

    Tensor<T> forward(Tensor<T> &input) override
    {
        auto output = softmax(input);

        auto gradNode = std::make_shared<SoftmaxBackward<T>>(input);
        output.setGradNode(gradNode);

        return output;
    }
};

} // namespace lava::nn
