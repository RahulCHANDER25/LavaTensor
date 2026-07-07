/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Storage
*/

#pragma once

#include <memory>
#include <stdexcept>
#include <vector>
#include "Device.hpp"

namespace lava {

template <typename T>
class Storage {
    public:
    Storage(size_t size, std::shared_ptr<Device<T>> device) : _size(size), _device(device)
    {
        if (_device->isCPU()) {
            _cpuData.resize(_size, T{0});
            _data = _cpuData.data();
        } else {
            _data = _device->allocate(_size);
        }
    }

    ~Storage()
    {
        if (_data && !_device->isCPU()) {
            _device->free(_data);
        }
    }

    T *data()
    {
        if (_device->isCPU()) {
            return _cpuData.data();
        }
        return _data;
    }

    const T *data() const
    {
        if (_device->isCPU()) {
            return _cpuData.data();
        }
        return _data;
    }

    std::vector<T> &datas()
    {
        if (!_device->isCPU()) {
            throw std::runtime_error("Cannot access datas() vector directly for non-CPU device.");
        }
        return _cpuData;
    }

    const std::vector<T> &datas() const
    {
        if (!_device->isCPU()) {
            throw std::runtime_error("Cannot access datas() vector directly for non-CPU device.");
        }
        return _cpuData;
    }

    size_t size() const
    {
        return _size;
    }

    std::shared_ptr<Device<T>> device() const
    {
        return _device;
    }

    void dispRaw() const
    {
        _device->dispRaw(data(), _size);
    }

    private:
    T *_data{nullptr};
    std::vector<T> _cpuData;
    size_t _size{0};
    std::shared_ptr<Device<T>> _device;
};

} // namespace lava
