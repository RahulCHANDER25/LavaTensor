/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ReLU
*/

#pragma once

#include <algorithm>
#include <memory>
#include "../Tensor/Tensor.hpp"
#include "Module.hpp"
#include "Tensor/autograd/ReLUBackward.hpp"

namespace lava::nn {

template <typename T>
class ReLU : public Module<T> {
    public:
    ReLU() = default;

    ~ReLU() override = default;

    Tensor<T> forward(Tensor<T> &input) override
    {
        Tensor<T> output({static_cast<int>(input.datas().size())});

        // ReLU forward: max(0, x)
        for (size_t i = 0; i < input.datas().size(); ++i) {
            output[i] = std::max(static_cast<T>(0), input[i]);
        }

        auto gradNode = std::make_shared<ReLUBackward<T>>(input);
        output.setGradNode(gradNode);

        return output;
    }
};

} // namespace lava::nn
