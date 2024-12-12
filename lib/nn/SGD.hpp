/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** SGD
*/

#pragma once

#include <cmath>
#include <memory>
#include <vector>
#include "Linear.hpp"
#include "Module.hpp"

namespace lava::nn {

template <typename T>
class SGD {
    public:
    SGD(const std::vector<std::shared_ptr<Module<T>>> &layers, T learningRate = 0.01)
        : _layers(layers), _learningRate(learningRate)
    {
    }

    void zeroGrad()
    {
        for (auto &layer : _layers) {
            if (auto *linear = dynamic_cast<Linear<T> *>(layer.get())) {
                auto &weightGrad = linear->_weights.grad().datas();
                std::fill(weightGrad.begin(), weightGrad.end(), 0);

                auto &biasGrad = linear->_biases.grad().datas();
                std::fill(biasGrad.begin(), biasGrad.end(), 0);
            }
        }
    }

    void step()
    {
        const T maxGrad = 1.0;

        for (auto &layer : _layers) {
            if (auto *linear = dynamic_cast<Linear<T> *>(layer.get())) {
                auto &weights = linear->_weights;
                auto &weightData = weights.tensor().datas();
                const auto &gradData = weights.grad().datas();

                for (size_t i = 0; i < weightData.size(); ++i) {
                    T grad = gradData[i];
                    if (std::isnan(grad) || std::isinf(grad)) {
                        continue;
                    }
                    grad = std::max(std::min(grad, maxGrad), -maxGrad);
                    weightData[i] -= _learningRate * grad;
                }

                auto &biases = linear->_biases;
                auto &biasData = biases.tensor().datas();
                const auto &biasGradData = biases.grad().datas();

                for (size_t i = 0; i < biasData.size(); ++i) {
                    T grad = biasGradData[i];
                    if (std::isnan(grad) || std::isinf(grad)) {
                        continue;
                    }
                    grad = std::max(std::min(grad, maxGrad), -maxGrad);
                    biasData[i] -= _learningRate * grad;
                }
            }
        }
    }

    private:
    const std::vector<std::shared_ptr<Module<T>>> &_layers;
    T _learningRate;
};

} // namespace lava::nn
