/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** main
*/

#include <iostream>
#include <memory>
#include <vector>
#include "ArgParser.hpp"
#include "ChessboardParser.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/TensorArray.hpp"
#include "Tensor/autograd/AddBackward.hpp"
#include "Tensor/autograd/MulBackward.hpp"
#include "Tensor/autograd/SubBackward.hpp"
#include "nn/Linear.hpp"
#include "nn/Sequential.hpp"
#include "training/chessTraining.hpp"

void printVec(std::vector<int> &vec)
{
    for (auto v: vec) {
        std::cout << v << std::endl;
    }
}

int main(int , char *[])
{
    lava::Tensor<double> x({2}, true);

    lava::nn::Linear<double> lin{2, 1};

    x.dispRaw();
    auto c = lin.forward(x); // Fix the matrix mul and for 1 dim and so on

    std::cout << "Res:\n";
    c.dispRaw();
    std::cout << "Backward:\n";
    c.backward();

    std::cout << "Grads:\n";
    lin._weights.grad().dispRaw();
    lin._biases.grad().dispRaw();
    // auto out = tensor1 * tensor2;

    // tensor1.dispRaw();
    // tensor2.dispRaw();

    // out.backward();
    // out.grad().dispRaw();
    // out.tensor().dispRaw();
    // tensor1.grad().dispRaw();
    // tensor2.grad().dispRaw();
    // try {
    //     auto args = ArgParser::parseAnalyzerArgs(argc, argv);
    //     {
    //         // TODO: #5 Implement neural network loading
    //         std::cout << "Loading neural network from: " << args.loadFile << std::endl;
    //     }

    //     auto boards = ChessboardParser::parseChessboardFile(args.inputFile);

    //     if (args.isPredictMode) {
    //         std::cout << "Running in prediction mode" << std::endl;
    //         for (const auto &board : boards) {
    //             {
    //                 // TODO: #4 Implement prediction logic
    //                 std::cout << "Analyzing position: " << board.fen << std::endl;
    //                 std::cout << "Nothing" << std::endl;
    //             }
    //         }
    //     } else if (args.isTrainMode) {
    //         // TODO: #3 Implement training logic
    //         lava::nn::Sequential<double> model{
    //             std::make_shared<lava::nn::Linear<double>>(lava::nn::Linear<double>(2, 3))
    //         };
    //         lava::train::chessTrain(model, boards);

    //         // for (const auto &board : boards) {
    //         //     std::cout << "Analyzing position: " << board.fen << " Size: " << board.boardData.size() << std::endl;
    //         //     for (size_t i = 0; i < board.boardData.size(); i++) {
    //         //         if (i != 0 && i % 8 == 0) {
    //         //             std::cout << " " << i << " ";
    //         //         }
    //         //         std::cout << ((int) board.boardData[i]);
    //         //     }
    //         //     std::cout << std::endl;
    //         //     std::cout << (board.expectedOutput.empty() ? "No label" : board.expectedOutput) << std::endl;
    //         //     std::cout << board.outLabel << std::endl << std::endl;
    //         // }
    //     }

    //     return 0;
    // } catch (const std::exception &e) {
    //     std::cerr << "Error: " << e.what() << std::endl;
    //     return 84;
    // }
}
