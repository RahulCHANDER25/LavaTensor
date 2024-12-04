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

    auto thisPtr = std::make_shared<Tensor>(*this);
    auto othPtr = std::make_shared<Tensor>(oth);

    auto gradFn = [thisPtr, othPtr](const Tensor<T> &grad) {
        if (thisPtr->_requiresGrad) {
            TensorArray<T> gradWrtThis = grad.tensor().matmul(othPtr->_tensor);
            thisPtr->_grad += gradWrtThis;
        }
        if (othPtr->_requiresGrad) {
            TensorArray<T> gradWrtOth = thisPtr->_tensor.transpose().matmul(grad.tensor());
            othPtr->_grad += gradWrtOth;
        }
    };
    auto gradNode = std::make_shared<MMBackward<T>>(*this, oth);

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

    // auto thisPtr = std::make_shared<Tensor>(*this);
    // auto othPtr = std::make_shared<Tensor>(oth);

    // auto gradFn = [thisPtr, othPtr](const Tensor<T> &grad) {
    //     // Derivative is 1 for this
    //     // Derivative is 1 for oth
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor(grad.tensor());
    //         thisPtr->_grad += gradTensor;
    //     }
    //     if (othPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor(grad.tensor());
    //         othPtr->_grad += gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);
    // auto othPtr = std::make_shared<Tensor>(oth);

    // auto gradFn = [thisPtr, othPtr](const Tensor<T> &grad) {
    //     // Derivative is 1 for this
    //     // Derivative is -1 for oth
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor(grad.tensor());
    //         thisPtr->_grad += gradTensor;
    //     }
    //     if (othPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor(grad.tensor());
    //         othPtr->_grad -= gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);
    // auto othPtr = std::make_shared<Tensor>(oth);

    // auto gradFn = [thisPtr, othPtr](const Tensor<T> &grad) {
    //     // derivative for this is oth
    //     // derivative for oth is this
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor = grad.tensor() * othPtr->_tensor;
    //         thisPtr->_grad += gradTensor;
    //     }
    //     if (othPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor = grad.tensor() * thisPtr->_tensor;
    //         othPtr->_grad += gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);
    // auto othPtr = std::make_shared<Tensor>(oth);

    // auto gradFn = [thisPtr, othPtr](const Tensor<T> &grad) {
    //     // derivative for this is 1/oth
    //     // derivative for oth is -this/(oth^2)
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor = grad.tensor() / othPtr->_tensor;
    //         thisPtr->_grad += gradTensor;
    //     }
    //     if (othPtr->_requiresGrad) {
    //         TensorArray<T> numerator = grad.tensor() * thisPtr->_tensor;
    //         TensorArray<T> denominator = othPtr->_tensor * othPtr->_tensor;
    //         TensorArray<T> gradTensor = numerator / denominator;
    //         othPtr->_grad -= gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);

    // auto gradFn = [thisPtr](const Tensor<T> &grad) {
    //     // derivative is 1
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor(grad.tensor());
    //         thisPtr->_grad += gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);

    // auto gradFn = [thisPtr](const Tensor<T> &grad) {
    //     // derivative is 1
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor(grad.tensor());
    //         thisPtr->_grad += gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);

    // auto gradFn = [thisPtr, k](const Tensor<T> &grad) {
    //     // derivative is k
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor = grad.tensor() * k;
    //         thisPtr->_grad += gradTensor;
    //     }
    // };

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

    // auto thisPtr = std::make_shared<Tensor>(*this);

    // auto gradFn = [thisPtr, k](const Tensor<T> &grad) {
    //     // derivative is 1/k
    //     if (thisPtr->_requiresGrad) {
    //         TensorArray<T> gradTensor = grad.tensor() / k;
    //         thisPtr->_grad += gradTensor;
    //     }
    // };

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

    this->_gradNode->backward();
    // if (!_requiresGrad) {
    //     return;
    // }

    // if (!_gradFn && _grad.datas().empty()) {
    //     _grad = TensorArray<T>(_tensor.shape(), _tensor.strides());
    //     std::fill(_grad.datas().begin(), _grad.datas().end(), T{1});
    // }

    // if (_gradFn) {
    //     _gradFn(*this);
    // }

    // for (const auto &prev : _previous) {
    //     if (prev && prev->requiresGrad()) {
    //         prev->backward();
    //     }
    // }
}

template <typename T>
void lava::Tensor<T>::zeroGrad()
{
    if (_requiresGrad) { // Not here
        std::fill(_grad.datas().begin(), _grad.datas().end(), T{0});
    }
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::createWithGrad( // this too
    TensorArray<T> data,
    std::vector<std::shared_ptr<Tensor<T>>> prev,
    BackwardFunction<T> gradFn
)
{
    Tensor result(data, true);
    result._previous = std::move(prev);
    result._gradFn = std::move(gradFn);
    return result;
}

template <typename T>
lava::Tensor<T> lava::Tensor<T>::createWithGrad(
    TensorArray<T> data,
    std::shared_ptr<GradNode<T>> gradNode
)
{
    return Tensor{data, gradNode, true};
}
