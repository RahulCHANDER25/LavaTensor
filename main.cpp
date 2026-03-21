#include "lib/Tensor/Tensor.hpp"

int main()
{
    lava::Tensor<float> a({2, 3});
    lava::Tensor<float> b({2, 3});

    auto c = a + b;

    std::cout << c.tensor().shape()[0] << " " << c.tensor().shape()[1] << std::endl;
    return 0;
}
