/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** GradNode
*/

#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include "Tensor/TensorArray.hpp"

namespace lava {

// ONE on init TensorArray
// By default leaf have Accumulate grad type

template <typename T>
class GradNode {
public:
    GradNode() = default;
    virtual ~GradNode() = default;

    virtual void backward(TensorArray<T> grad) = 0;

    virtual void backward() = 0;

    void addNextGrad(std::shared_ptr<GradNode<T>> nextNode)
    {
        _nextGrads.push_back(nextNode);
    }

    std::vector<std::shared_ptr<GradNode<T>>> &getNextNodes() { return _nextGrads; }

    virtual void auth() { std::cout << "I am gradnode\n"; }

protected:
    std::vector<std::shared_ptr<GradNode<T>>> _nextGrads;
};

}