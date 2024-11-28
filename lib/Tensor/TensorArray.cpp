/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** TensorArray
*/

#include "TensorArray.hpp"

#include <format>
#include <iostream>
#include <stdexcept>

template <typename T>
void lava::TensorArray<T>::dispRaw()
{
    std::cout << "Datas:\n";
    for (const auto &elem: _datas) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

template <typename T>
lava::TensorArray<T>::TensorArray(std::initializer_list<int> shape):
    _shape(shape),
    _datas()
{
    size_t size = 1;

    for (const auto &s: _shape) {
        size *= s;
    }
    _datas.reserve(size);

    for (size_t k = 0; k < _shape.size(); k++) {
        _strides.push_back(getStride(k, _shape));
    }
    for (size_t i = 0; i < size; i++) {
        _datas.push_back(random() % 2);
    }
}

template <typename T>
lava::TensorArray<T>::TensorArray(std::initializer_list<int> shape, InitType type):
    _shape(shape),
    _datas()
{
    size_t size = 1;

    for (const auto &s: _shape) {
        size *= s;
    }
    _datas.reserve(size);

    for (size_t k = 0; k < _shape.size(); k++) {
        _strides.push_back(getStride(k, _shape));
    }
    if (type == InitType::RANDOM) {
        for (size_t i = 0; i < size; i++) {
            _datas.push_back(random() % 2);
        }
    }
    if (type == InitType::ZERO) {
        for (size_t i = 0; i < size; i++) {
            _datas.push_back(T{0});
        } 
    }
}

template <typename T>
lava::TensorArray<T>::TensorArray(const TensorArray &tensor):
    _shape(tensor._shape),
    _strides(tensor._strides),
    _datas(tensor._datas)
{
}

template <typename T>
lava::TensorArray<T>::TensorArray(const std::vector<int> &shape, const std::vector<int> &strides):
    _shape(shape),
    _strides(strides),
    _datas()
{
    size_t size = 1;

    for (const auto &s: _shape) {
        size *= s;
    }
    for (size_t i = 0; i < size; i++) {
        _datas.push_back(T{0});
    }
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::_inPlaceTensorOperation(
    const TensorArray &oth,
    std::function<T (const T &, const T &)> func
)
{
    for (size_t i = 0; i < _datas.size(); i++) {
        this->operator[](i) = func(this->operator[](i), oth[i]);
    }
    return *this;
}

template <typename T>
lava::TensorArray<T> lava::TensorArray<T>::_tensorOperation(
    const TensorArray &oth,
    std::function<T (const T &, const T &)> func
)
{
    TensorArray newTensor(_shape, _strides);

    for (size_t i = 0; i < _datas.size(); i++) {
        newTensor[i] = func(this->operator[](i), oth[i]);
    }
    return newTensor;
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::_inPlaceScalarOperation(T k, std::function<T (const T &, const T &)> func)
{
    for (size_t i = 0; i < _datas.size(); i++) {
        this->operator[](i) = func(this->operator[](i), k);
    }
    return *this;
}

template <typename T>
lava::TensorArray<T> lava::TensorArray<T>::_scalarOperation(
    T k,
    std::function<T (const T &, const T &)> func
)
{
    TensorArray<T> newTensor(_shape, _strides);

    for (size_t i = 0; i < _datas.size(); i++) {
        newTensor[i] = func(this->operator[](i), k);
    }
    return newTensor;
}

template <typename T>
lava::TensorArray<T> lava::TensorArray<T>::matmul(const TensorArray &oth) // only 2 DIM Tensors are supported
{
    if (oth._shape.size() != _shape.size() && _shape.size() != 2) {
        throw std::logic_error("[ERR] Only 2 Dimensional Tensors are supported for matmul");
    }
    if (_shape[1] != oth._shape[0]) {
        throw std::logic_error("Incorrect dimension for the matrix multiplication.");
    }
    TensorArray<T> newTensor({_shape[0], oth._shape[1]}, TensorArray::InitType::ZERO);

    for (int i = 0; i < _shape[0]; i++) {
        for (int j = 0; j < _shape[1]; j++) {
            for (int k = 0; k < oth._shape[1]; k++) { // Here we can change some things for gradient
                newTensor({i, k}) += this->operator()({i, j}) * oth.operator()({j, k});
            }
        }
    }
    return newTensor;
}

template <typename T>
T lava::TensorArray<T>::operator[](size_t idx) const
{
    if (idx >= _datas.size()) {
        throw std::out_of_range(
            std::format("[ERR]: Index {} is out of range of tensor of size {}.", idx, _datas.size())
        );
    }
    return _datas[idx];
}

template <typename T>
T &lava::TensorArray<T>::operator[](size_t idx)
{
    if (idx >= _datas.size()) {
        throw std::out_of_range(
            std::format("[ERR]: Index {} is out of range of tensor of size {}.", idx, _datas.size())
        );
    }
    return _datas[idx];
}

template <typename T>
T lava::TensorArray<T>::operator()(std::initializer_list<int> indexes) const
{
    if (indexes.size() != _strides.size()) {
        throw std::logic_error("[ERR] Incorrect number of dimensions given.");
    }

    int idx = 0;
    const auto *itIndexes = indexes.begin();
    for (int stride : _strides) {
        idx += (*itIndexes) * stride;
        itIndexes++;
    }
    return _datas[idx];
}

template <typename T>
T& lava::TensorArray<T>::operator()(std::initializer_list<int> indexes)
{
    if (indexes.size() != _strides.size()) {
        throw std::logic_error("[ERR] Incorrect number of dimensions given.");
    }

    int idx = 0;
    const auto *itIndexes = indexes.begin();
    for (size_t k = 0; k < _strides.size(); k++) {
        if ((*itIndexes) >= _shape[k]) {
            throw std::out_of_range("[ERR] Out of range using tensor indexes.");
        }
        idx += (*itIndexes) * _strides[k];
        itIndexes++;
    }
    return _datas[idx];
}

/**
 *  Static Functions below.
 */

template <typename T>
size_t lava::TensorArray<T>::getStride(int k, const std::vector<int> &shape)
{
    int stride = 1;

    for (size_t j = k + 1; j < shape.size(); j++) {
        stride *= shape[j];
    }
    return stride;
}
