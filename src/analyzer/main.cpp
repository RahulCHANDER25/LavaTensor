/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** main
*/

#include <iostream>
#include <vector>
#include "ArgParser.hpp"
#include "ChessboardParser.hpp"
#include "nn/Sequential.hpp"
#include "training/chessTraining.hpp"
#include "utils/NetworkConfig.hpp"
#include "utils/NetworkLoader.hpp"

std::vector<std::string> predictPositions(
    lava::nn::Sequential<double> &model,
    const std::vector<ChessboardParser::ChessboardData> &boards
)
{
    std::vector<std::string> predictions;
    const std::vector<std::string> classes = {
        "Checkmate White", "Checkmate Black", "Check White", "Check Black", "Stalemate", "Nothing"
    };

    for (const auto &board : boards) {
        std::vector<int> inputShape = {1, static_cast<int>(board.boardData.size())};
        std::vector<int> strides = {static_cast<int>(board.boardData.size()), 1}; // Row-major strides
        lava::TensorArray<double> tensorArray(inputShape, strides);
        tensorArray.datas() = board.boardData;
        lava::Tensor<double> input(tensorArray);
        auto output = model.forward(input);

        size_t predictedClass = 0;
        const auto &outputData = output.tensor().datas();
        double maxProb = outputData[0];
        for (size_t i = 1; i < outputData.size(); i++) {
            if (outputData[i] > maxProb) {
                maxProb = outputData[i];
                predictedClass = i;
            }
        }

        predictions.push_back(classes[predictedClass]);
    }

    return predictions;
}

int main(int argc, char *argv[])
{
    try {
        auto args = ArgParser::parseAnalyzerArgs(argc, argv);

        std::cout << "Loading neural network from: " << args.loadFile << std::endl;
        auto model = lava::NetworkLoader::loadNetwork(args.loadFile);
        auto boards = ChessboardParser::parseChessboardFile(args.inputFile);

        if (args.isPredictMode) {
            std::cout << "Running in prediction mode" << std::endl;
            auto predictions = predictPositions(*model, boards);
            for (const auto &pred : predictions) {
                std::cout << pred << std::endl;
            }
        } else if (args.isTrainMode) {
            std::cout << "Running in training mode" << std::endl;

            lava::train::TrainingConfig config;
            config.shouldSave = !args.saveFile.empty();
            config.saveFile = args.saveFile.empty() ? args.loadFile : args.saveFile;

            auto networkConfig = lava::NetworkConfig::fromFile("examples/basic_network.conf");
            config.learningRate = networkConfig.hyperparameters().learningRate;
            config.batchSize = networkConfig.hyperparameters().batchSize;
            config.epochs = networkConfig.hyperparameters().epochs;
            config.samplesPerEpoch = networkConfig.hyperparameters().samplesPerEpoch;

            // Load learning rate scheduler configuration
            const auto& lrScheduler = networkConfig.lrScheduler();
            config.schedulerType = lrScheduler.type;
            config.decayRate = lrScheduler.decayRate;
            config.decaySteps = lrScheduler.decaySteps;
            config.minLearningRate = lrScheduler.minLR;

            lava::train::chessTrain(*model, boards, config);
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 84;
    }
}
