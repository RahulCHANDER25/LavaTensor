/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Linear
*/

#pragma once

#include <cmath>
#include "Module.hpp"
#include "Tensor/Tensor.hpp"

namespace lava::nn {

template <typename T>
class Linear : public Module<T> {
public:
    Linear(int inFeatures, int outFeatures):
        _weights({inFeatures, outFeatures}, true),
        _biases({outFeatures, 1}, true)
    {
    }

    ~Linear() override = default;

    Tensor<T> forward(Tensor<T>& x) override
    {
        return x.matmul(this->_weights) + _biases;
    }

    // Tensor<T>& backward(Tensor<T>& gradOutput) override
    // {
    //     auto& inputArray = _lastInput.array();
    //     auto& inputData = inputArray.datas();
    //     std::vector<int> transposedShape = {inputArray.shape()[1], inputArray.shape()[0]};
    //     TensorArray<T> transposedArray(transposedShape, {1, transposedShape[1]});
    //     auto& transposedData = transposedArray.datas();
    //     for (int i = 0; i < inputArray.shape()[0]; ++i) {
    //         for (int j = 0; j < inputArray.shape()[1]; ++j) {
    //             transposedData[j * transposedShape[1] + i] = inputData[i * inputArray.shape()[1] + j];
    //         }
    //     }
    //     Tensor<T> inputTranspose(transposedArray, false);

    //     auto& weightsArray = _weights.tensor().array();
    //     auto& weightsData = weightsArray.datas();
    //     std::vector<int> transposedWeightsShape = {weightsArray.shape()[1], weightsArray.shape()[0]};
    //     TensorArray<T> transposedWeightsArray(transposedWeightsShape, {1, transposedWeightsShape[1]});
    //     auto& transposedWeightsData = transposedWeightsArray.datas();
    //     for (int i = 0; i < weightsArray.shape()[0]; ++i) {
    //         for (int j = 0; j < weightsArray.shape()[1]; ++j) {
    //             transposedWeightsData[j * transposedWeightsShape[1] + i] = weightsData[i * weightsArray.shape()[1] + j];
    //         }
    //     }
    //     Tensor<T> weightsTranspose(transposedWeightsArray, false);

    //     _weights.grad() = inputTranspose.matmul(gradOutput);
    //     _biases.grad() = gradOutput;
    //     _gradInput = gradOutput.matmul(weightsTranspose);

    //     return _gradInput;
    // }

    // Only weights and biases as Tensor
    Tensor<T> _weights;
    Tensor<T> _biases;
private:
//     Parameter<T> _weights;
//     Parameter<T> _biases;

// private:
//     Tensor<T> _lastInput;
//     Tensor<T> _gradInput;
};

}
