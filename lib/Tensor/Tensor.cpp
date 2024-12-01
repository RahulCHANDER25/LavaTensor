/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Tensor
*/

#include "Tensor.hpp"
#include "Tensor/TensorArray.hpp"

template <typename T>
lava::Tensor<T>::Tensor(std::initializer_list<int> shape):
    _tensor(shape),
    _grad(shape)
{
}

template <typename T>
lava::Tensor<T>::Tensor(const Tensor<T> &tensor):
    _tensor(tensor._tensor),
    _grad(tensor._grad)
{
}

template <typename T>
lava::Tensor<T>::Tensor(const TensorArray<T> &tensor):
    _tensor(tensor),
    _grad(tensor.shape(), tensor.strides())
{
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::matmul(const Tensor &oth)
{
    auto newTensor = _tensor.matmul(oth._tensor);

    // I think we should redo the matmul for this case specifically (not very hard I think)
    return Tensor<T>(newTensor);
}

template <typename T>
lava::Tensor<T> &lava::Tensor<T>::operator=(const lava::Tensor<T> &oth)
{
    this->_tensor = oth._tensor;
    this->_grad = oth._grad;
    return *this;
}

template <typename T>
lava::Tensor<T> &lava::Tensor<T>::operator=(lava::Tensor<T> &&oth) noexcept
{
    this->_tensor = std::move(oth._tensor);
    this->_grad = std::move(oth._grad);
    return *this;
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::Tensor::operator+(Tensor &oth)
{
    // Derivative is 1 for this
    // Derivative is 1 for oth
    return {_tensor + oth._tensor};
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::Tensor::operator-(Tensor &oth)
{
    // Derivative is 1 for this
    // Derivative is 1 for oth
    return {_tensor - oth._tensor};  
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::Tensor::operator*(Tensor &oth)
{
    // Derivative is oth._tensor for this
    // Derivative is this for oth
    return {_tensor * oth._tensor};
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::Tensor::operator/(Tensor &oth)
{
    // Derivative is 1/oth._tensor for this
    // Derivative is -_tensor/oth._tensor for oth ==> To verify ??
    return {_tensor / oth._tensor};
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator+(T k)
{
    // Derivative 1 for this
    return {_tensor + k};
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator-(T k)
{
    // Derivative 1 for this
    return {_tensor - k};
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator*(T k)
{
    // The partial derivative here is the k factor filling a whole Tensor the same size as the `this`
    return {_tensor * k};
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator/(T k)
{
    // The partial derivative here is the 1/k factor filling a whole Tensor the same size as the `this`
    return {_tensor / k};
}

template <typename T>
T lava::Tensor<T>::operator()(std::initializer_list<int> indexes) const
{
    return _tensor(std::move(indexes));
}

template <typename T>
T &lava::Tensor<T>::operator()(std::initializer_list<int> indexes)
{
    return _tensor(std::move(indexes));
}

template <typename T>
T lava::Tensor<T>::operator[](size_t idx) const
{
    return _tensor[idx];
}

template <typename T>
T &lava::Tensor<T>::operator[](size_t idx)
{
    return _tensor[idx];
}

