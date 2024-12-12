/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** NetworkGenerator
*/

#pragma once

#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <random>
#include "nn/Linear.hpp"
#include "nn/Module.hpp"
#include "nn/ReLU.hpp"
#include "nn/Softmax.hpp"
#include "utils/NetworkConfig.hpp"

namespace lava {

class NetworkGenerator {
    public:
    static void generateNetwork(const NetworkConfig &config, const std::string &outputPath)
    {
        std::ofstream file(outputPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not create network file: " + outputPath);
        }

        Header header{};
        std::memcpy(header.magic, MAGIC, 4);
        header.version = VERSION;
        header.archHash = computeArchHash(config);
        header.numLayers =
            config.architecture().hiddenLayers * 2 + 2; // Linear + ReLU for each hidden layer + output + softmax
        std::memset(header.reserved, 0, sizeof(header.reserved));

        file.write(reinterpret_cast<const char *>(&header), sizeof(header));

        auto layers = generateLayers(config);
        writeLayers(file, layers);

        file.close();
    }

    private:
    static constexpr char MAGIC[] = "LAVA";
    static constexpr uint32_t VERSION = 1;

    struct Header {
        char magic[4];
        uint32_t version;
        uint64_t archHash;
        uint32_t numLayers;
        char reserved[12];
    };

    enum class LayerType : uint32_t {
        LINEAR = 1,
        RELU = 2,
        SOFTMAX = 3
    };

    struct LayerHeader {
        LayerType type;
        uint32_t inputSize;
        uint32_t outputSize;
        uint32_t activation;
    };

    static std::vector<std::shared_ptr<nn::Module<double>>> generateLayers(const NetworkConfig &config)
    {
        std::vector<std::shared_ptr<nn::Module<double>>> layers;
        const auto &arch = config.architecture();
        const auto &init = config.initialization();

        size_t prevSize = arch.inputSize;
        for (size_t size : arch.hiddenSizes) {
            layers.push_back(std::make_shared<nn::Linear<double>>(prevSize, size));
            layers.push_back(std::make_shared<nn::ReLU<double>>());
            prevSize = size;
        }

        layers.push_back(std::make_shared<nn::Linear<double>>(prevSize, arch.outputSize));
        layers.push_back(std::make_shared<nn::Softmax<double>>());

        initializeWeights(layers, init);
        return layers;
    }

    static void initializeWeights(
        std::vector<std::shared_ptr<nn::Module<double>>> &layers,
        const NetworkConfig::Initialization &init
    )
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (auto &layer : layers) {
            if (auto linear = std::dynamic_pointer_cast<nn::Linear<double>>(layer)) {
                switch (init.weightInit) {
                    case WeightInit::XAVIER: {
                        auto &weightData = linear->_weights.tensor().datas();
                        double limit = std::sqrt(
                            6.0 / (linear->_weights.tensor().shape()[0] + linear->_weights.tensor().shape()[1])
                        );
                        std::uniform_real_distribution<double> dist(-limit, limit);
                        for (auto &w : weightData) {
                            w = dist(gen);
                        }
                        break;
                    }
                    case WeightInit::HE: {
                        auto &weightData = linear->_weights.tensor().datas();
                        double stddev = std::sqrt(2.0 / linear->_weights.tensor().shape()[0]);
                        std::normal_distribution<double> dist(0.0, stddev);
                        for (auto &w : weightData) {
                            w = dist(gen);
                        }
                        break;
                    }
                    case WeightInit::UNIFORM: {
                        auto &weightData = linear->_weights.tensor().datas();
                        std::uniform_real_distribution<double> dist(-1.0, 1.0);
                        for (auto &w : weightData) {
                            w = dist(gen);
                        }
                        break;
                    }
                }

                switch (init.biasInit) {
                    case BiasInit::ZEROS: {
                        auto &biasData = linear->_biases.tensor().datas();
                        std::fill(biasData.begin(), biasData.end(), 0.0);
                        break;
                    }
                    case BiasInit::UNIFORM: {
                        auto &biasData = linear->_biases.tensor().datas();
                        std::uniform_real_distribution<double> dist(-1.0, 1.0);
                        for (auto &b : biasData) {
                            b = dist(gen);
                        }
                        break;
                    }
                }
            }
        }
    }

    static void writeLayers(std::ofstream &file, const std::vector<std::shared_ptr<nn::Module<double>>> &layers)
    {
        for (const auto &layer : layers) {
            if (auto linear = std::dynamic_pointer_cast<nn::Linear<double>>(layer)) {
                writeLinearLayer(file, linear);
            } else if (auto relu = std::dynamic_pointer_cast<nn::ReLU<double>>(layer)) {
                writeReluLayer(file);
            } else if (auto softmax = std::dynamic_pointer_cast<nn::Softmax<double>>(layer)) {
                writeSoftmaxLayer(file);
            }
        }
    }

    static void writeLinearLayer(std::ofstream &file, const std::shared_ptr<nn::Linear<double>> &layer)
    {
        LayerHeader header{
            LayerType::LINEAR,
            static_cast<uint32_t>(layer->_weights.tensor().shape()[0]),
            static_cast<uint32_t>(layer->_weights.tensor().shape()[1]),
            0
        };

        file.write(reinterpret_cast<const char *>(&header), sizeof(header));

        const auto &weights = layer->_weights.tensor().datas();
        file.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(double));

        const auto &biases = layer->_biases.tensor().datas();
        file.write(reinterpret_cast<const char *>(biases.data()), biases.size() * sizeof(double));
    }

    static void writeReluLayer(std::ofstream &file)
    {
        LayerHeader header{LayerType::RELU, 0, 0, 0};
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }

    static void writeSoftmaxLayer(std::ofstream &file)
    {
        LayerHeader header{LayerType::SOFTMAX, 0, 0, 0};
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }

    static uint64_t computeArchHash(const NetworkConfig &config)
    {
        const auto &arch = config.architecture();
        uint64_t hash = arch.inputSize;
        for (auto size : arch.hiddenSizes) {
            hash = hash * 31 + size;
        }
        hash = hash * 31 + arch.outputSize;
        return hash;
    }
};

} // namespace lava
