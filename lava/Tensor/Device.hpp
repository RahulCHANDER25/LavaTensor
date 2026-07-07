/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Device
*/

#pragma once

#include <cstddef>
#include <vector>

namespace lava {

template <typename T>
class Device {
    public:
    virtual ~Device() = default;

    // Device identification
    virtual bool isCPU() const = 0;

    // Memory operations
    virtual T *allocate(size_t count) = 0;
    virtual void free(T *ptr) = 0;
    virtual void copyHostToDevice(T *dst, const T *src, size_t count) = 0;
    virtual void copyDeviceToHost(T *dst, const T *src, size_t count) = 0;

    // Element-wise operations (C = A op B)
    virtual void add(const T *a, const T *b, T *c, size_t size) = 0;
    virtual void sub(const T *a, const T *b, T *c, size_t size) = 0;
    virtual void mul(const T *a, const T *b, T *c, size_t size) = 0;
    virtual void div(const T *a, const T *b, T *c, size_t size) = 0;

    // Scalar operations (C = A op k)
    virtual void addScalar(const T *a, T k, T *c, size_t size) = 0;
    virtual void subScalar(const T *a, T k, T *c, size_t size) = 0;
    virtual void mulScalar(const T *a, T k, T *c, size_t size) = 0;
    virtual void divScalar(const T *a, T k, T *c, size_t size) = 0;

    // Strided Matrix Multiplication (C = A x B)
    virtual void matmul(
        const T *a,
        const std::vector<int> &aShape,
        const std::vector<int> &aStrides,
        const T *b,
        const std::vector<int> &bShape,
        const std::vector<int> &bStrides,
        T *c,
        const std::vector<int> &cShape,
        const std::vector<int> &cStrides
    ) = 0;

    // Display the raw data of the tensor (for debugging)
    virtual void dispRaw(const T *data, size_t size) = 0;
};

} // namespace lava
