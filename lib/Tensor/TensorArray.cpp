/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** TensorArray
*/

#include "TensorArray.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include <initializer_list>

template <typename T>
void lava::TensorArray<T>::dispRaw()
{
    std::cout << "Datas:\n";
    for (const auto &elem : _datas) {
        std::cout << elem << " ";
    }
    std::cout << std::endl;
}

template <typename T>
lava::TensorArray<T>::TensorArray(std::initializer_list<int> shape, InitType type) : _shape(shape), _datas()
{
    size_t size = 1;

    for (const auto &s : _shape) {
        size *= s;
    }
    _datas.reserve(size);

    for (size_t k = 0; k < _shape.size(); k++) {
        _strides.push_back(getStride(k, _shape));
    }
    if (type == InitType::RANDOM) {
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_floating_point_v<T>) {
            // He initialization for floating point types
            T stddev = static_cast<T>(std::sqrt(2.0 / _shape[0]));
            std::normal_distribution<T> dist(0.0, stddev);
            for (size_t i = 0; i < size; i++) {
                _datas.push_back(dist(gen));
            }
        } else {
            // For integer types, use a uniform distribution
            // Scale based on input size similar to He init
            int range = static_cast<int>(std::sqrt(6.0 / _shape[0]));
            std::uniform_int_distribution<int> dist(-range, range);
            for (size_t i = 0; i < size; i++) {
                _datas.push_back(static_cast<T>(dist(gen)));
            }
        }
    }
    if (type == InitType::ZERO) {
        for (size_t i = 0; i < size; i++) {
            _datas.push_back(T{0});
        }
    }
    if (type == InitType::ONES) {
        for (size_t i = 0; i < size; i++) {
            _datas.push_back(1);
        }
    }
    if (type == InitType::RANGE) {
        for (size_t i = 0; i < size; i++) {
            _datas.push_back(i);
        }
    }
}

template <typename T>
lava::TensorArray<T>::TensorArray(const TensorArray &tensor)
    : _shape(tensor._shape), _strides(tensor._strides), _datas(tensor._datas)
{
}

template <typename T>
lava::TensorArray<T>::TensorArray(TensorArray &&tensor) noexcept
    : _shape(std::move(tensor._shape)), _strides(std::move(tensor._strides)), _datas(std::move(tensor._datas))
{
}

template <typename T>
lava::TensorArray<T>::TensorArray(const std::vector<T> &datas)
    : _shape(std::initializer_list<int>{(int)datas.size()}), _strides(1), _datas(datas)
{
}

template <typename T>
lava::TensorArray<T>::TensorArray(const std::vector<int> &shape, const std::vector<int> &strides)
    : _shape(shape), _strides(strides), _datas()
{
    size_t size = 1;

    for (const auto &s : _shape) {
        size *= s;
    }
    for (size_t i = 0; i < size; i++) {
        _datas.push_back(T{0});
    }
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::_inPlaceTensorOperation(
    const TensorArray &oth,
    std::function<T(const T &, const T &)> func
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
    std::function<T(const T &, const T &)> func
) const
{
    TensorArray newTensor(_shape, _strides);

    for (size_t i = 0; i < _datas.size(); i++) {
        newTensor[i] = func(this->operator[](i), oth[i]);
    }
    return newTensor;
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::_inPlaceScalarOperation(T k, std::function<T(const T &, const T &)> func)
{
    for (size_t i = 0; i < _datas.size(); i++) {
        this->operator[](i) = func(this->operator[](i), k);
    }
    return *this;
}

template <typename T>
lava::TensorArray<T> lava::TensorArray<T>::_scalarOperation(T k, std::function<T(const T &, const T &)> func) const
{
    TensorArray<T> newTensor(_shape, _strides);

    for (size_t i = 0; i < _datas.size(); i++) {
        newTensor[i] = func(this->operator[](i), k);
    }
    return newTensor;
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::unsqueezed(size_t dim)
{
    if (dim > _shape.size()) {
        throw std::logic_error("[ERR] Scalar Product not supported yet");
    }
    _shape.insert(_shape.begin() + dim, 1);
    _strides.push_back(1);
    for (size_t k = 0; k < _shape.size(); k++) {
        _strides[k] = getStride(k, _shape);
    }
    return *this;
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::removeDim(size_t dim)
{
    if (dim > _shape.size()) {
        throw std::logic_error("[ERR] Scalar Product not supported yet");
    }
    _shape.erase(_shape.begin() + dim);
    _strides.pop_back();
    for (size_t k = 0; k < _shape.size(); k++) {
        _strides[k] = getStride(k, _shape);
    }
    return *this;
}

template <typename T>
lava::TensorArray<T> lava::TensorArray<T>::matmul(TensorArray &oth) // only 2 DIM Tensors are supported
{
    if (oth.shape().size() == 1 && _shape.size() == 1) {
        throw std::logic_error("[ERR] Scalar Product not supported yet");
    }
    bool isUnsqueezed = false;
    if (_shape.size() == 1) {
        isUnsqueezed = true;
        unsqueezed();
    }
    if (oth._shape.size() != _shape.size()) {
        throw std::logic_error("[ERR] Only 2 Dimensional Tensors are supported for matmul");
    }
    if (_shape[1] != oth._shape[0]) {
        throw std::logic_error("Incorrect dimension for the matrix multiplication.");
    }
    TensorArray<T> newTensor({_shape[0], oth._shape[1]}, TensorArray::InitType::ZERO);

    for (int i = 0; i < _shape[0]; i++) {
        for (int j = 0; j < _shape[1]; j++) {
            for (int k = 0; k < oth._shape[1]; k++) {
                newTensor({i, k}) += this->operator()({i, j}) * oth.operator()({j, k});
            }
        }
    }
    if (isUnsqueezed) {
        removeDim();
    }
    return newTensor;
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::operator=(lava::TensorArray<T> &&oth) noexcept
{
    this->_datas = std::move(oth._datas);
    this->_shape = std::move(oth._shape);
    this->_strides = std::move(oth._strides);
    return *this;
}

template <typename T>
lava::TensorArray<T> lava::TensorArray<T>::transpose() const
{
    if (_shape.size() != 2) {
        throw std::logic_error("[ERR] Only 2 Dimensional Tensors are supported for transpose");
    }

    std::vector<int> newShape(_shape.size());
    std::vector<int> newStrides(_strides.size());
    std::reverse_copy(_shape.begin(), _shape.end(), newShape.begin());
    std::reverse_copy(_strides.begin(), _strides.end(), newStrides.begin());

    TensorArray<T> result(newShape, newStrides);

    for (int i = 0; i < _shape[0]; i++) {
        for (int j = 0; j < _shape[1]; j++) {
            result({i, j}) = this->operator()({i, j});
        }
    }
    return result;
}

template <typename T>
lava::TensorArray<T> &lava::TensorArray<T>::transposed()
{
    if (_shape.size() != 2) {
        throw std::logic_error("[ERR] Only 2 Dimensional Tensors are supported for transpose");
    }

    std::reverse(_shape.begin(), _shape.end());
    std::reverse(_strides.begin(), _strides.end());
    return *this;
}

template <typename T>
T lava::TensorArray<T>::operator[](size_t idx) const
{
    if (idx >= _datas.size()) {
        throw std::out_of_range(std::format("[ERR]: Index {} is out of range of tensor of size {}.", idx, _datas.size())
        );
    }
    return _datas[idx];
}

template <typename T>
T &lava::TensorArray<T>::operator[](size_t idx)
{
    if (idx >= _datas.size()) {
        throw std::out_of_range(std::format("[ERR]: Index {} is out of range of tensor of size {}.", idx, _datas.size())
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
T &lava::TensorArray<T>::operator()(std::initializer_list<int> indexes)
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

template <typename T>
size_t lava::TensorArray<T>::getStride(size_t k, const std::vector<int> &shape)
{
    if (k == (shape.size() - 1)) {
        return 1;
    }
    size_t stride = 1;

    for (size_t j = k + 1; j < shape.size(); j++) {
        stride *= shape[j];
    }
    return stride;
}
