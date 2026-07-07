#include "Tensor/Tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

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

int main()
{
    std::cout << "Starting LavaTensor Unit Tests..." << std::endl;
    std::cout << "----------------------------------" << std::endl;

    test_tensor_creation();
    test_tensor_addition();
    test_simple_autograd();

    std::cout << "----------------------------------" << std::endl;
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
