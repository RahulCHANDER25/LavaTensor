/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** chessTraining
*/

#include "training/chessTraining.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include "Tensor/TensorArray.hpp"
#include "nn/CrossEntropyLoss.hpp"
#include "nn/SGD.hpp"
#include "nn/Sequential.hpp"
#include "utils/NetworkSaver.hpp"

namespace lava::train {

size_t getLabelIndex(const std::string &labelStr)
{
    if (labelStr.find("Checkmate") != std::string::npos) {
        return (labelStr.find("White") != std::string::npos) ? 0 : 1;
    } else if (labelStr.find("Check") != std::string::npos) {
        return (labelStr.find("White") != std::string::npos) ? 2 : 3;
    } else if (labelStr.find("Stalemate") != std::string::npos) {
        return 4;
    }
    return 5; // Nothing
}

void chessTrain(
    nn::Module<double> &net,
    const std::vector<ChessboardParser::ChessboardData> &datas,
    const TrainingConfig &config
)
{
    nn::CrossEntropyLoss<double> criterion;
    auto *sequential = dynamic_cast<nn::Sequential<double> *>(&net);
    if (!sequential) {
        throw std::runtime_error("Network must be Sequential");
    }
    nn::SGD<double> optimizer(sequential->layers(), config.learningRate);

    // Create shuffled indices for batching
    std::vector<size_t> indices(datas.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());

    std::cout << "\nStarting training with " << datas.size() << " samples" << std::endl;
    std::cout << "Training Configuration:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Learning rate: " << config.learningRate << std::endl;
    std::cout << "Batch size: " << config.batchSize << std::endl;
    std::cout << "Number of epochs: " << config.epochs << std::endl;
    std::cout << "Save file: " << (config.saveFile.empty() ? "none" : config.saveFile) << std::endl;
    std::cout << "Should save: " << (config.shouldSave ? "yes" : "no") << std::endl;
    std::cout << "----------------------" << std::endl;

    std::cout << "\nNetwork Architecture:" << std::endl;
    std::cout << "----------------------" << std::endl;
    for (size_t i = 0; i < sequential->layers().size(); ++i) {
        const auto &layer = sequential->layers()[i];
        if (auto linear = std::dynamic_pointer_cast<nn::Linear<double>>(layer)) {
            std::cout << "Layer " << i << ": Linear(in=" << linear->_weights.tensor().shape()[0]
                      << ", out=" << linear->_weights.tensor().shape()[1] << ")" << std::endl;
        } else if (std::dynamic_pointer_cast<nn::ReLU<double>>(layer)) {
            std::cout << "Layer " << i << ": ReLU" << std::endl;
        } else if (std::dynamic_pointer_cast<nn::Softmax<double>>(layer)) {
            std::cout << "Layer " << i << ": Softmax" << std::endl;
        }
    }
    std::cout << "----------------------" << std::endl;

    for (size_t epoch = 0; epoch < config.epochs; epoch++) {
        double epochLoss = 0.0;
        size_t correct = 0;

        std::shuffle(indices.begin(), indices.end(), gen);

        for (size_t i = 0; i < indices.size(); i += config.batchSize) {
            size_t batchSize = std::min(config.batchSize, indices.size() - i);
            double batchLoss = 0.0;
            optimizer.zeroGrad();
            for (size_t j = 0; j < batchSize; j++) {
                size_t idx = indices[i + j];
                const auto &board = datas[idx];

                std::vector<int> inputShape = {1, static_cast<int>(board.boardData.size())};
                std::vector<int> strides = {static_cast<int>(board.boardData.size()), 1};
                std::vector<double> normalizedData = board.boardData;

                lava::TensorArray<double> tensorArray(inputShape, strides);
                tensorArray.datas() = normalizedData;
                Tensor<double> input(tensorArray);
                auto output = net.forward(input);

                size_t labelIndex = getLabelIndex(board.expectedOutput);

                double loss = criterion.forward(output, labelIndex);
                batchLoss += loss;

                size_t predictedClass = 0;
                const auto &outputData = output.tensor().datas();
                double maxProb = outputData[0];
                for (size_t k = 1; k < outputData.size(); k++) {
                    if (outputData[k] > maxProb) {
                        maxProb = outputData[k];
                        predictedClass = k;
                    }
                }

                if (predictedClass == labelIndex) {
                    correct++;
                }

                TensorArray<double> gradArray({static_cast<int>(outputData.size())}, {1});
                std::fill(gradArray.datas().begin(), gradArray.datas().end(), 0.0);
                Tensor<double> gradOutput(gradArray);
                gradOutput = criterion.backward(gradOutput);

                auto &gradData = gradOutput.tensor().datas();
                for (auto &grad : gradData) {
                    grad /= batchSize;
                    if (std::isnan(grad) || std::isinf(grad)) {
                        grad = 0.0;
                    }
                }

                net.backward(gradOutput);
            }

            optimizer.step();

            epochLoss += batchLoss / batchSize; // Average loss for the batch
        }

        double accuracy = static_cast<double>(correct) / datas.size();
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " - Loss: " << std::fixed << std::setprecision(4)
                  << epochLoss * config.batchSize / datas.size() << " - Accuracy: " << std::fixed
                  << std::setprecision(2) << accuracy * 100 << "%" << std::endl;

        if (config.shouldSave && !config.saveFile.empty() && (epoch + 1) % 10 == 0) // Save every 10 epochs
        {
            NetworkSaver::saveNetwork(
                std::shared_ptr<nn::Sequential<double>>(sequential, [](nn::Sequential<double> *) {}), config.saveFile
            );
            std::cout << "Checkpoint saved to " << config.saveFile << std::endl;
        }
    }

    std::cout << "\nTraining completed!" << std::endl;
}

} // namespace lava::train
