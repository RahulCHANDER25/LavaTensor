/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** chessTraining
*/

#pragma once

#include <vector>
#include "ChessboardParser.hpp"
#include "nn/Module.hpp"

namespace lava::train {

void chessTrain(
    lava::nn::Module<double> &net,
    const std::vector<ChessboardParser::ChessboardData> &datas
);

}
