/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Sequential
*/

#pragma once

#include <memory>
#include "Tensor/Tensor.hpp"
#include "nn/Module.hpp"
#include <initializer_list>

namespace lava::nn {

template <typename T>
class Sequential : public Module<T> { // Careful maybe again template specification on types
public:
    Sequential(std::initializer_list<std::shared_ptr<Module<T>>> modules):
        _modules(modules)
    {}

    ~Sequential() override = default;

    Tensor<T> forward(const Tensor<T> &in) override
    {
        Tensor<T> out{in};
        for (auto &mod: _modules) {
            out = mod->forward(out);
        }
        return std::move(out);
    }

private:
    std::vector<std::shared_ptr<Module<T>>> _modules;
};

}
