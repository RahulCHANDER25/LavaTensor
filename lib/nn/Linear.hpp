/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Linear
*/

#pragma once

#include <random>
#include <cmath>
#include "Module.hpp"
#include "Parameter.hpp"

namespace lava::nn {

template <typename T>
class Linear : public Module<T> {
public:
    Linear(int inFeatures, int outFeatures)
        : _weights({inFeatures, outFeatures}, {outFeatures, 1}),
          _biases({outFeatures}, {1}),
          _lastInput(TensorArray<T>({1, 1}, {1}), false),
          _gradInput(TensorArray<T>({1, 1}, {1}), false)
    {
        auto& weightData = _weights.tensor().datas();
        T stddev = std::sqrt(2.0 / inFeatures);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<T> dist(0.0, stddev);
        for (auto& w : weightData) {
            w = dist(gen);
        }

        auto& biasData = _biases.tensor().datas();
        std::fill(biasData.begin(), biasData.end(), 0.0);
    }

    ~Linear() override = default;

    Tensor<T> forward(const Tensor<T>& input) override
    {
        TensorArray<T> inputArray = input.array();
        Tensor<T> inputCopy(inputArray, false);
        _lastInput = inputCopy;  // Save for backward

        auto result = inputCopy.matmul(_weights.tensor());
        return result + _biases.tensor();
    }

    Tensor<T>& backward(Tensor<T>& gradOutput) override
    {
        auto& inputArray = _lastInput.array();
        auto& inputData = inputArray.datas();
        std::vector<int> transposedShape = {inputArray.shape()[1], inputArray.shape()[0]};
        TensorArray<T> transposedArray(transposedShape, {1, transposedShape[1]});
        auto& transposedData = transposedArray.datas();
        for (int i = 0; i < inputArray.shape()[0]; ++i) {
            for (int j = 0; j < inputArray.shape()[1]; ++j) {
                transposedData[j * transposedShape[1] + i] = inputData[i * inputArray.shape()[1] + j];
            }
        }
        Tensor<T> inputTranspose(transposedArray, false);

        auto& weightsArray = _weights.tensor().array();
        auto& weightsData = weightsArray.datas();
        std::vector<int> transposedWeightsShape = {weightsArray.shape()[1], weightsArray.shape()[0]};
        TensorArray<T> transposedWeightsArray(transposedWeightsShape, {1, transposedWeightsShape[1]});
        auto& transposedWeightsData = transposedWeightsArray.datas();
        for (int i = 0; i < weightsArray.shape()[0]; ++i) {
            for (int j = 0; j < weightsArray.shape()[1]; ++j) {
                transposedWeightsData[j * transposedWeightsShape[1] + i] = weightsData[i * weightsArray.shape()[1] + j];
            }
        }
        Tensor<T> weightsTranspose(transposedWeightsArray, false);

        _weights.grad() = inputTranspose.matmul(gradOutput);
        _biases.grad() = gradOutput;
        _gradInput = gradOutput.matmul(weightsTranspose);

        return _gradInput;
    }

    Parameter<T> _weights;
    Parameter<T> _biases;

private:
    Tensor<T> _lastInput;
    Tensor<T> _gradInput;
};

}
