/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Tensor
*/

#pragma once

#include "Tensor/TensorArray.hpp"

namespace lava {

template <typename T>
class Tensor {
public:
    void dispRaw()
    {
        dprintf(1, "Hello I am a Tensor\n");
    }

    // Decorate all the operators (operator+, operator-, matmul, ....)

    Tensor matmul(const Tensor &oth);

    Tensor operator+(Tensor &oth) { return _tensorOperation(oth, std::plus<T>()); }
    Tensor operator-(Tensor &oth) { return _tensorOperation(oth, std::minus<T>()); }
    Tensor operator*(Tensor &oth) { return _tensorOperation(oth, std::multiplies<T>()); }
    Tensor operator/(Tensor &oth)
    {
        return _tensorOperation(
            oth,
            [] (const T &a, const T &b) {
                if (b == 0) {
                    throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
                }
                return std::divides<T>()(a, b);
            }
        );
    }

private:
    TensorArray<T> _tensor; /** TensorArray with the tensor datas */
    TensorArray<T> _grad; /** TensorArray with the tensor gradient => Maybe std::optionnal for non-leaf and inference mode ? */

    // std::shared_ptr<BackwardFunc> _node;
    /** Node (BackwardFunc associated) of the current tensor for the backpropagation
        => No node if it is not a leaf, but the backwardFunc is still added to the graph
    */
};

}
