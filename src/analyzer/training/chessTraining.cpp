/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** chessTraining
*/

#include "training/chessTraining.hpp"
#include <iostream>
#include "Tensor/TensorArray.hpp"

void lava::train::chessTrain(
    lava::nn::Module<double> &net,
    const std::vector<ChessboardParser::ChessboardData> &datas
)
{
    for (const auto &board : datas) {
        Tensor<double> input(TensorArray<double>{board.boardData});
        std::cout << board.fen << " " << std::endl;
        auto out = net.forward(input);

        out.dispRaw();
        out.backward();

        // break;
        // With result compute error
        // backward on error
        // apply grad on layers // if there is an optimizer this is a step
    }
}
