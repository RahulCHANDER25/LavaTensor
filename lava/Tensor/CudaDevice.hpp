/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** CudaDevice
*/

#pragma once

#include "Device.hpp"
#include <memory>
#include <vector>

namespace lava {

template <typename T>
class CudaDevice : public Device<T> {
public:
    CudaDevice();
    ~CudaDevice() override;

    bool isCPU() const override { return false; }

    T* allocate(size_t count) override;
    void free(T* ptr) override;
    
    void copyHostToDevice(T* dst, const T* src, size_t count) override;
    void copyDeviceToHost(T* dst, const T* src, size_t count) override;

    void add(const T* a, const T* b, T* c, size_t size) override;
    void sub(const T* a, const T* b, T* c, size_t size) override;
    void mul(const T* a, const T* b, T* c, size_t size) override;
    void div(const T* a, const T* b, T* c, size_t size) override;

    void addScalar(const T* a, T k, T* c, size_t size) override;
    void subScalar(const T* a, T k, T* c, size_t size) override;
    void mulScalar(const T* a, T k, T* c, size_t size) override;
    void divScalar(const T* a, T k, T* c, size_t size) override;

    void matmul(
        const T* a, const std::vector<int>& aShape, const std::vector<int>& aStrides,
        const T* b, const std::vector<int>& bShape, const std::vector<int>& bStrides,
        T* c, const std::vector<int>& cShape, const std::vector<int>& cStrides
    ) override;

    void dispRaw(const T *data, size_t size) override;
};

} // namespace lava
