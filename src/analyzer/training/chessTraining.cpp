/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** chessTraining
*/

#include "training/chessTraining.hpp"
#include <iostream>

void lava::train::chessTrain(
    lava::nn::Module<double> &/* net */,
    const std::vector<ChessboardParser::ChessboardData> &datas
)
{
    for (const auto &board : datas) {
        std::cout << board.fen << " " << board.fen << std::endl;
        // net.forward();
        
        // With result compute error
        // backward on error
        // apply grad on layers // if there is an optimizer this is a step
    }
}
