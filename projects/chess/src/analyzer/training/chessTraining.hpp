/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** chessTraining
*/

#pragma once

#include <string>
#include <vector>
#include "ChessboardParser.hpp"
#include "nn/Module.hpp"
#include "nn/Sequential.hpp"

namespace lava::train {

struct TrainingConfig {
    size_t epochs{100};
    double learningRate{0.1};
    size_t batchSize{32};
    size_t samplesPerEpoch{10000};
    std::string saveFile;
    bool shouldSave{false};
    std::string schedulerType{"none"};
    double decayRate{1.0};
    size_t decaySteps{100};
    double minLearningRate{0.0001};
};

void trainSummary(
    const std::vector<ChessboardParser::ChessboardData> &datas,
    const TrainingConfig &config
);

void networkSummary(lava::nn::Sequential<double> *sequential);

void chessTrain(
    lava::nn::Module<double> &net,
    const std::vector<ChessboardParser::ChessboardData> &datas,
    const TrainingConfig &config = TrainingConfig{}
);

} // namespace lava::train
