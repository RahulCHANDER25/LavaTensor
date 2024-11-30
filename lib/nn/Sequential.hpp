/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Sequential
*/

#pragma once

#include "nn/Module.hpp"
#include <initializer_list>

namespace lava::nn {

template <typename T>
class Sequential : public Module<T> { // Careful maybe again template specification on types
public:
    Sequential(std::initializer_list<Module<T>> modules):
        _modules(modules)
    {}

    ~Sequential() override = default;

    Tensor<T> forward(const Tensor<T> &) override
    {
        // Overload operator=()
        for (const auto &mod: _modules) {
            // mod.forward()
        }
        return {2, 3};
    }

private:
    std::vector<Module<T>> _modules;
};

}
