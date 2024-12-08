/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Tensor
*/

#include "Tensor.hpp"
#include <memory>
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/AccumulateBackward.hpp"
#include "Tensor/autograd/AddBackward.hpp"
#include "Tensor/autograd/DivBackward.hpp"
#include "Tensor/autograd/GradNode.hpp"
#include "Tensor/autograd/MMBackward.hpp"
#include "Tensor/autograd/MulBackward.hpp"
#include "Tensor/autograd/SubBackward.hpp"
#include "Tensor/autograd/SumBackward.hpp"

template <typename T>
lava::Tensor<T>::Tensor(std::initializer_list<int> shape) : _tensor(shape), _grad(shape)
{
}

template <typename T>
lava::Tensor<T>::Tensor(const Tensor<T> &tensor) : _tensor(tensor._tensor), _grad(tensor._grad)
{
}

template <typename T>
lava::Tensor<T>::Tensor(const TensorArray<T> &data, bool requiresGrad)
    : _tensor(std::move(data)), _grad(data.shape(), data.strides()), _requiresGrad(requiresGrad)
{
    if (requiresGrad) {
        _gradNode = std::make_shared<AccumulateBackward<T>>(*this);
        zeroGrad();
    }
}

template <typename T>
lava::Tensor<T>::Tensor(const TensorArray<T> &data, std::shared_ptr<GradNode<T>> gradNode, bool requiresGrad)
    : _tensor(std::move(data)), _grad(data.shape(), data.strides()), _requiresGrad(requiresGrad), _gradNode(gradNode)
{
    if (requiresGrad) {
        zeroGrad();
    }
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::matmul(Tensor &oth)
{
    TensorArray<T> result = _tensor.matmul(oth._tensor);

    if (!_requiresGrad && !oth._requiresGrad) {
        return Tensor(result, false);
    }

    bool isUnsqueezed = false;
    if (this->_tensor.shape().size() == 1) {
        isUnsqueezed = true;
        _tensor.unsqueezed();
    }
    auto gradNode = std::make_shared<MMBackward<T>>(*this, oth);

    if (isUnsqueezed) {
        _tensor.removeDim();
    }
    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::sum()
{
    T sumVal{0};

    for (const auto &val: _tensor.datas()) {
        sumVal += val;
    }
    TensorArray<T> arr{1};
    arr[0] = sumVal;
    auto gradNode = std::make_shared<SumBackward<T>>(*this);
    
    return createWithGrad(arr, gradNode);
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
    TensorArray<T> result = _tensor + oth._tensor;

    if (!_requiresGrad && !oth._requiresGrad) {
        return Tensor(result, false);
    }

    auto gradNode = std::make_shared<AddBackward<T>>(*this, oth);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator-(Tensor &oth)
{
    TensorArray<T> result = _tensor - oth._tensor;

    if (!_requiresGrad && !oth._requiresGrad) {
        return Tensor(result, false);
    }

    auto gradNode = std::make_shared<SubBackward<T>>(*this, oth);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator*(Tensor &oth)
{
    TensorArray<T> result = _tensor * oth._tensor;

    if (!_requiresGrad && !oth._requiresGrad) {
        return Tensor(result, false);
    }
    auto gradNode = std::make_shared<MulBackward<T>>(*this, oth);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator/(Tensor &oth) // Does not work correctly on gradient
{
    TensorArray<T> result = _tensor / oth._tensor;

    if (!_requiresGrad && !oth._requiresGrad) {
        return Tensor(result, false);
    }
    auto gradNode = std::make_shared<DivBackward<T>>(*this, oth);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator+(T k)
{
    TensorArray<T> result = _tensor + k;

    if (!_requiresGrad) {
        return Tensor(result, false);
    }
    auto gradNode = std::make_shared<AddBackward<T>>(*this);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator-(T k)
{
    TensorArray<T> result = _tensor - k;

    if (!_requiresGrad) {
        return Tensor(result, false);
    }
    auto gradNode = std::make_shared<SubBackward<T>>(*this);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator*(T k)
{
    TensorArray<T> result = _tensor * k;

    if (!_requiresGrad) {
        return Tensor(result, false);
    }
    auto gradNode = std::make_shared<MulBackward<T>>(*this, k);

    return createWithGrad(result, gradNode);
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::operator/(T k)
{
    TensorArray<T> result = _tensor / k;

    if (!_requiresGrad) {
        return Tensor(result, false);
    }
    auto gradNode = std::make_shared<DivBackward<T>>(*this, k);

    return createWithGrad(result, gradNode);
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

template <typename T>
void lava::Tensor<T>::backward() // Modify this
{
    // Check of dim 1
    // this->_gradNode->backward(); // From one dim to the input tensor

    if (_gradNode) {
        this->_gradNode->backward();
    }
}

template <typename T>
void lava::Tensor<T>::zeroGrad()
{
    if (_requiresGrad) { // Not here
        std::fill(_grad.datas().begin(), _grad.datas().end(), T{0});
    }
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::createWithGrad(
    TensorArray<T> data,
    std::shared_ptr<GradNode<T>> gradNode
)
{
    return Tensor{data, gradNode, true};
}
