/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** CudaDevice
*/

#include <iostream>
#include <stdexcept>
#include "CudaDevice.hpp"
#include <cuda_runtime.h>

// CUDA global kernels for element-wise operations
template <typename T>
__global__ void add_kernel(const T *a, const T *b, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

template <typename T>
__global__ void sub_kernel(const T *a, const T *b, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - b[idx];
    }
}

template <typename T>
__global__ void mul_kernel(const T *a, const T *b, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

template <typename T>
__global__ void div_kernel(const T *a, const T *b, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (b[idx] == T{0}) {
            // Note: In CUDA, standard practice is to handle zero division by producing infinity/NaN
            // or we could check, but raising runtime exceptions inside standard GPU kernels is generally not supported.
            c[idx] = T{0}; // Fallback or NaN-like representation depending on float vs int
        } else {
            c[idx] = a[idx] / b[idx];
        }
    }
}

// CUDA global kernels for scalar operations
template <typename T>
__global__ void add_scalar_kernel(const T *a, T k, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + k;
    }
}

template <typename T>
__global__ void sub_scalar_kernel(const T *a, T k, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] - k;
    }
}

template <typename T>
__global__ void mul_scalar_kernel(const T *a, T k, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * k;
    }
}

template <typename T>
__global__ void div_scalar_kernel(const T *a, T k, T *c, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] / k;
    }
}

// Strided matrix multiplication GPU kernel
template <typename T>
__global__ void matmul_strided_kernel(
    const T *a,
    int a_s0,
    int a_s1,
    const T *b,
    int b_s0,
    int b_s1,
    T *c,
    int c_s0,
    int c_s1,
    int M,
    int N,
    int K
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        T sum = 0;
        for (int j = 0; j < N; ++j) {
            sum += a[row * a_s0 + j * a_s1] * b[j * b_s0 + col * b_s1];
        }
        c[row * c_s0 + col * c_s1] = sum;
    }
}

namespace lava {

template <typename T>
CudaDevice<T>::CudaDevice()
{
}

template <typename T>
CudaDevice<T>::~CudaDevice()
{
}

template <typename T>
T *CudaDevice<T>::allocate(size_t count)
{
    T *devPtr = nullptr;
    cudaError_t err = cudaMalloc(&devPtr, count * sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    return devPtr;
}

template <typename T>
void CudaDevice<T>::free(T *ptr)
{
    cudaFree(ptr);
}

template <typename T>
void CudaDevice<T>::copyHostToDevice(T *dst, const T *src, size_t count)
{
    cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy HostToDevice failed: ") + cudaGetErrorString(err));
    }
}

template <typename T>
void CudaDevice<T>::copyDeviceToHost(T *dst, const T *src, size_t count)
{
    cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy DeviceToHost failed: ") + cudaGetErrorString(err));
    }
}

template <typename T>
void CudaDevice<T>::add(const T *a, const T *b, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::sub(const T *a, const T *b, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    sub_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::mul(const T *a, const T *b, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::div(const T *a, const T *b, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    div_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::addScalar(const T *a, T k, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    add_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, k, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::subScalar(const T *a, T k, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    sub_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, k, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::mulScalar(const T *a, T k, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    mul_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, k, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::divScalar(const T *a, T k, T *c, size_t size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    div_scalar_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, k, c, size);
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::matmul(
    const T *a,
    const std::vector<int> &aShape,
    const std::vector<int> &aStrides,
    const T *b,
    const std::vector<int> &bShape,
    const std::vector<int> &bStrides,
    T *c,
    const std::vector<int> &cShape,
    const std::vector<int> &cStrides
)
{
    int M = aShape[0];
    int N = aShape[1];
    int K = bShape[1];

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + 15) / 16, (M + 15) / 16);

    matmul_strided_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        a, aStrides[0], aStrides[1], b, bStrides[0], bStrides[1], c, cStrides[0], cStrides[1], M, N, K
    );
    cudaDeviceSynchronize();
}

template <typename T>
void CudaDevice<T>::dispRaw(const T *data, size_t size)
{
    std::vector<T> hostData(size);
    copyDeviceToHost(hostData.data(), data, size);
    for (size_t i = 0; i < size; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;
}

// Explicit instantiations
template class CudaDevice<int>;
template class CudaDevice<size_t>;
template class CudaDevice<double>;
template class CudaDevice<float>;

} // namespace lava
