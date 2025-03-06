#include <torch/torch.h>
#include <iostream>

int main() {
    // Print a message to verify that PyTorch is being used
    std::cout << "PyTorch is working!" << std::endl;

    // Create a tensor filled with random values
    torch::Tensor tensor = torch::randn({ 2, 3 });  // 2x3 matrix with random values
    std::cout << "Tensor: " << tensor << std::endl;

    // Check tensor's properties
    std::cout << "Tensor size: " << tensor.sizes() << std::endl;

    return 0;
}
