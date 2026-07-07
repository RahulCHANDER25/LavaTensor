#include "Tensor/Tensor.hpp"

int main()
{
    // Create two 2D tensors
    lava::Tensor<float> tensorA({2, 3}, true);
    lava::Tensor<float> tensorB({3, 4}, true);

    // Perform matrix multiplication
    lava::Tensor<float> result = tensorA.matmul(tensorB);

    // Display the result
    result.dispRaw();

    return 0;
}
