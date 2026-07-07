/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** CPUDevice
*/

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "Device.hpp"

namespace lava {

template <typename T>
class CPUDevice : public Device<T> {
    public:
    CPUDevice() = default;
    ~CPUDevice() override = default;

    bool isCPU() const override
    {
        return true;
    }

    T *allocate(size_t count) override
    {
        return new T[count];
    }

    void free(T *ptr) override
    {
        delete[] ptr;
    }

    void copyHostToDevice(T *dst, const T *src, size_t count) override
    {
        std::copy(src, src + count, dst);
    }

    void copyDeviceToHost(T *dst, const T *src, size_t count) override
    {
        std::copy(src, src + count, dst);
    }

    void add(const T *a, const T *b, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void sub(const T *a, const T *b, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] - b[i];
        }
    }

    void mul(const T *a, const T *b, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] * b[i];
        }
    }

    void div(const T *a, const T *b, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            if (b[i] == T{0}) {
                throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
            }
            c[i] = a[i] / b[i];
        }
    }

    void addScalar(const T *a, T k, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] + k;
        }
    }

    void subScalar(const T *a, T k, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] - k;
        }
    }

    void mulScalar(const T *a, T k, T *c, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] * k;
        }
    }

    void divScalar(const T *a, T k, T *c, size_t size) override
    {
        if (k == T{0}) {
            throw std::logic_error("[ERR] Zero division Error while doing a div operation.");
        }
        for (size_t i = 0; i < size; ++i) {
            c[i] = a[i] / k;
        }
    }

    void matmul(
        const T *a,
        const std::vector<int> &aShape,
        const std::vector<int> &aStrides,
        const T *b,
        const std::vector<int> &bShape,
        const std::vector<int> &bStrides,
        T *c,
        const std::vector<int> &cShape,
        const std::vector<int> &cStrides
    ) override
    {
        int M = aShape[0];
        int N = aShape[1];
        int K = bShape[1];

        // Initialize output matrix to zero
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                c[i * cStrides[0] + k * cStrides[1]] = T{0};
            }
        }

        // Perform strided matrix multiplication
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                T valA = a[i * aStrides[0] + j * aStrides[1]];
                for (int k = 0; k < K; ++k) {
                    c[i * cStrides[0] + k * cStrides[1]] += valA * b[j * bStrides[0] + k * bStrides[1]];
                }
            }
        }
    }

    void dispRaw(const T *data, size_t size) override
    {
        for (size_t i = 0; i < size; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

} // namespace lava
