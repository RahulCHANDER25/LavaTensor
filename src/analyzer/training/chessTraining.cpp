/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** chessTraining
*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

#include "Tensor/TensorArray.hpp"
#include "nn/CrossEntropyLoss.hpp"
#include "nn/SGD.hpp"
#include "nn/Sequential.hpp"
#include "training/chessTraining.hpp"
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

void trainSummary(const std::vector<ChessboardParser::ChessboardData> &datas, const TrainingConfig &config)
{
    std::cout << "\nStarting training with " << datas.size() << " total samples" << std::endl;
    std::cout << "Training Configuration:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "Initial learning rate: " << config.learningRate << std::endl;
    std::cout << "Batch size: " << config.batchSize << std::endl;
    std::cout << "Samples per epoch: " << config.samplesPerEpoch << std::endl;
    std::cout << "Number of epochs: " << config.epochs << std::endl;
    std::cout << "Save file: " << (config.saveFile.empty() ? "none" : config.saveFile) << std::endl;
    std::cout << "Should save: " << (config.shouldSave ? "yes" : "no") << std::endl;
    if (config.schedulerType != "none") {
        std::cout << "Learning rate scheduler: " << config.schedulerType << std::endl;
        std::cout << "Decay rate: " << config.decayRate << std::endl;
        std::cout << "Decay steps: " << config.decaySteps << std::endl;
        std::cout << "Minimum learning rate: " << config.minLearningRate << std::endl;
    }
    std::cout << "----------------------" << std::endl;
}

void networkSummary(lava::nn::Sequential<double> *sequential) // In nn.Module
{
    std::cout << "\nNetwork Architecture:" << std::endl;
    std::cout << "----------------------" << std::endl;
    for (size_t i = 0; i < sequential->layers().size(); ++i) {
        const auto &layer = sequential->layers()[i];
        if (auto linear = std::dynamic_pointer_cast<nn::Linear<double>>(layer)) {
            std::cout << "Layer " << i << ": Linear(in=" << linear->_weights.tensor().shape()[0]
                      << ", out=" << linear->_weights.shape()[1] << ")" << std::endl;
        } else if (std::dynamic_pointer_cast<nn::ReLU<double>>(layer)) {
            std::cout << "Layer " << i << ": ReLU" << std::endl;
        } else if (std::dynamic_pointer_cast<nn::Softmax<double>>(layer)) {
            std::cout << "Layer " << i << ": Softmax" << std::endl;
        }
    }
    std::cout << "----------------------" << std::endl;
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

    // Create indices for the entire dataset
    std::vector<size_t> allIndices(datas.size());
    std::iota(allIndices.begin(), allIndices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());

    trainSummary(datas, config);
    networkSummary(sequential);

    const unsigned int numThreads = std::thread::hardware_concurrency();
    const size_t samplesPerEpoch = std::min(config.samplesPerEpoch, datas.size());

    for (size_t epoch = 0; epoch < config.epochs; epoch++) {
        // Update learning rate if scheduler is enabled
        if (config.schedulerType == "exponential") {
            double newLR =
                config.learningRate * std::pow(config.decayRate, static_cast<double>(epoch) / config.decaySteps);
            newLR = std::max(newLR, config.minLearningRate);
            optimizer.setLearningRate(newLR);
        }

        double epochLoss = 0.0;
        std::atomic<size_t> correct{0};

        // Standard shuffle without execution policy
        std::shuffle(allIndices.begin(), allIndices.end(), gen);

        // Create epoch indices (subset of shuffled indices)
        std::vector<size_t> epochIndices(allIndices.begin(), allIndices.begin() + samplesPerEpoch);

        // Process batches
        for (size_t i = 0; i < samplesPerEpoch; i += config.batchSize) {
            size_t batchSize = std::min(config.batchSize, samplesPerEpoch - i);
            std::atomic<double> batchLoss{0.0};
            optimizer.zeroGrad();

            // Parallel processing of batch samples
            std::vector<std::future<void>> futures;
            size_t chunkSize = std::max(size_t(1), batchSize / numThreads);

            for (size_t start = 0; start < batchSize; start += chunkSize) {
                size_t end = std::min(start + chunkSize, batchSize);
                futures.push_back(std::async(std::launch::async, [&, start, end]() {
                    double localLoss = 0.0;
                    size_t localCorrect = 0;

                    for (size_t j = start; j < end; j++) {
                        const auto &board = datas[epochIndices[i + j]];

                        std::vector<int> inputShape = {1, static_cast<int>(board.boardData.size())};
                        std::vector<int> strides = {static_cast<int>(board.boardData.size()), 1};
                        std::vector<double> normalizedData = board.boardData;

                        lava::TensorArray<double> tensorArray(inputShape, strides);
                        tensorArray.datas() = normalizedData;
                        Tensor<double> input(tensorArray);

                        auto output = net.forward(input);
                        size_t labelIndex = getLabelIndex(board.expectedOutput);
                        size_t predictedClass = output.argmax();

                        auto loss = criterion.forward(output, labelIndex);
                        loss.backward();

                        localLoss += loss[0];
                        if (predictedClass == labelIndex) {
                            localCorrect++;
                        }
                    }

                    batchLoss += localLoss;
                    correct += localCorrect;
                }));
            }

            // Wait for all threads to complete
            for (auto &future : futures) {
                future.wait();
            }

            optimizer.step();
            epochLoss += static_cast<double>(batchLoss) / batchSize;
        }

        double accuracy = static_cast<double>(correct) / samplesPerEpoch;
        std::cout << "Epoch " << epoch + 1 << "/" << config.epochs << " (" << samplesPerEpoch
                  << " samples) - Loss: " << std::fixed << std::setprecision(4)
                  << epochLoss * config.batchSize / samplesPerEpoch << " - Accuracy: " << std::fixed
                  << std::setprecision(2) << accuracy * 100 << "% - LR: " << std::scientific << std::setprecision(3)
                  << optimizer.getLearningRate() << std::endl;

        if (config.shouldSave && !config.saveFile.empty() && (epoch + 1) % 10 == 0) {
            NetworkSaver::saveNetwork(
                std::shared_ptr<nn::Sequential<double>>(sequential, [](nn::Sequential<double> *) {}), config.saveFile
            );
            std::cout << "Checkpoint saved to " << config.saveFile << std::endl;
        }
    }

    std::cout << "\nTraining completed!" << std::endl;
}

} // namespace lava::train
