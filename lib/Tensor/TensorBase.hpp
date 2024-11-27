/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** TensorBase
*/

#pragma once

namespace lava {

class TensorBase {
public:
    virtual ~TensorBase() = default;

    virtual void dispRaw() = 0;
    // Function utils as virtual
};

}
