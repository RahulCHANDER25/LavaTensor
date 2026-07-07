# LavaTensor

LavaTensor is a lightweight C++ multidimensional array and neural network library designed from scratch. It features standard linear algebra operations and an automatic differentiation (autograd) engine that constructs dynamic computational graphs for backpropagation.

## Features

- **Multidimensional Tensors**: Strides-based layout supporting operations like matrix multiplication, transposing, addition, division, and slicing.
- **Autograd Engine**: Automatic reverse-mode differentiation (`GradNode` computational graph tracking) to calculate gradients for machine learning optimization.
- **Neural Network Library**: Standard layers (`Linear`, `ReLU`, `Softmax`), loss functions (`CrossEntropyLoss`), and optimizers (`SGD`).
- **Heterogeneous Architecture Ready**: Structured to support clean transitions between CPU and GPU hardware backends.

## Quick Start

```cpp
#include "Tensor/Tensor.hpp"
#include <iostream>

int main() {
    // Create tensors with autograd enabled
    lava::TensorArray<float> xData({2.0f});
    lava::Tensor<float> x(xData, true); // requiresGrad = true

    // Compute y = x * 3.0
    auto y = x * 3.0f;

    // Propagate gradients backwards through the computation graph
    y.backward();

    // Access the accumulated gradient (dy/dx = 3.0)
    std::cout << "Gradient dy/dx: " << x.grad()[0] << std::endl;

    return 0;
}
```

## Repository Structure

```
LavaTensor/
├── CMakeLists.txt              # Root CMake configuration
├── lava/                       # Core LavaTensor library
│   ├── Tensor/                 # Tensor and TensorArray definitions
│   │   └── autograd/           # Dynamic computational graph nodes
│   └── nn/                     # Neural Network module and layers
├── projects/
│   └── chess/                  # Chess position prediction and training application
└── tests/                      # Core library unit tests
```

## Build and Run

LavaTensor uses CMake for cross-platform building.

### Prerequisites

- CMake 3.20 or higher
- C++23 compliant compiler (e.g., GCC 13+ or Clang 16+)

### Compilation

```bash
# Configure the build system
cmake -B build

# Build all targets (library, chess apps, and tests)
cmake --build build
```

### Running Tests

To verify that the library and autograd engine function correctly, run the unit test suite:

```bash
./build/tests/lava_tests
```
