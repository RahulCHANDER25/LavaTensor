#include "Tensor/Tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

#ifdef LAVA_HAS_CUDA
#include "Tensor/CudaDevice.hpp"
#endif

void test_tensor_creation()
{
    std::cout << "[RUN] test_tensor_creation..." << std::endl;

    lava::Tensor<float> a({2, 3});
    assert(a.shape().size() == 2);
    assert(a.shape()[0] == 2);
    assert(a.shape()[1] == 3);
    assert(a.datas().size() == 6);

    std::cout << "[PASS] test_tensor_creation" << std::endl;
}

void test_tensor_addition()
{
    std::cout << "[RUN] test_tensor_addition..." << std::endl;

    lava::TensorArray<float> arrA(std::vector<float>{1.5f, 2.5f, 3.5f});
    lava::TensorArray<float> arrB(std::vector<float>{0.5f, 1.5f, 2.5f});

    lava::Tensor<float> a(arrA);
    lava::Tensor<float> b(arrB);

    auto c = a + b;

    assert(c.shape().size() == 1);
    assert(c.shape()[0] == 3);
    assert(std::abs(c[0] - 2.0f) < 1e-5);
    assert(std::abs(c[1] - 4.0f) < 1e-5);
    assert(std::abs(c[2] - 6.0f) < 1e-5);

    std::cout << "[PASS] test_tensor_addition" << std::endl;
}

void test_simple_autograd()
{
    std::cout << "[RUN] test_simple_autograd..." << std::endl;

    lava::TensorArray<float> xData(std::vector<float>{2.0f});
    lava::Tensor<float> x(xData, true); // requiresGrad = true

    // y = x * 3.0
    auto y = x * 3.0f;
    y.backward();

    // Gradient dy/dx should be 3.0
    assert(std::abs(x.grad()[0] - 3.0f) < 1e-5);

    std::cout << "[PASS] test_simple_autograd" << std::endl;
}

#ifdef LAVA_HAS_CUDA
void test_cuda_addition()
{
    std::cout << "[RUN] test_cuda_addition..." << std::endl;

    auto cudaDev = std::make_shared<lava::CudaDevice<float>>();
    lava::TensorArray<float> arrA(std::vector<float>{1.5f, 2.5f, 3.5f}, cudaDev);
    lava::TensorArray<float> arrB(std::vector<float>{0.5f, 1.5f, 2.5f}, cudaDev);

    lava::TensorArray<float> c = arrA + arrB;

    std::vector<float> hostResult(3);
    cudaDev->copyDeviceToHost(hostResult.data(), c.storage()->data(), 3);

    assert(std::abs(hostResult[0] - 2.0f) < 1e-5);
    assert(std::abs(hostResult[1] - 4.0f) < 1e-5);
    assert(std::abs(hostResult[2] - 6.0f) < 1e-5);

    std::cout << "[PASS] test_cuda_addition" << std::endl;
}

void test_cuda_matmul()
{
    std::cout << "[RUN] test_cuda_matmul..." << std::endl;

    auto cudaDev = std::make_shared<lava::CudaDevice<float>>();
    // A: 2x3
    lava::TensorArray<float> arrA(std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }, cudaDev);
    arrA.shape() = {2, 3};
    arrA.strides() = {3, 1};

    // B: 3x2
    lava::TensorArray<float> arrB(std::vector<float>{
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    }, cudaDev);
    arrB.shape() = {3, 2};
    arrB.strides() = {2, 1};

    lava::TensorArray<float> c = arrA.matmul(arrB);

    std::vector<float> hostResult(4);
    cudaDev->copyDeviceToHost(hostResult.data(), c.storage()->data(), 4);

    // C = A * B
    // [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12] = [58,  64]
    // [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]   [139, 154]
    assert(std::abs(hostResult[0] - 58.0f) < 1e-5);
    assert(std::abs(hostResult[1] - 64.0f) < 1e-5);
    assert(std::abs(hostResult[2] - 139.0f) < 1e-5);
    assert(std::abs(hostResult[3] - 154.0f) < 1e-5);

    std::cout << "[PASS] test_cuda_matmul" << std::endl;
}
#endif

int main()
{
    std::cout << "Starting LavaTensor Unit Tests..." << std::endl;
    std::cout << "----------------------------------" << std::endl;

    test_tensor_creation();
    test_tensor_addition();
    test_simple_autograd();

#ifdef LAVA_HAS_CUDA
    test_cuda_addition();
    test_cuda_matmul();
#endif

    std::cout << "----------------------------------" << std::endl;
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
