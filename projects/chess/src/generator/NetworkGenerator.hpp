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
            config.architecture().hiddenLayers * 2 + 1; // Linear + ReLU for each hidden layer + output + softmax
        std::memset(header.reserved, 0, sizeof(header.reserved));

        file.write(reinterpret_cast<const char *>(&header), sizeof(header));

        writeConfiguration(file, config);

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
    } __attribute__((packed));

    struct ConfigHeader {
        uint32_t hyperparamsSize;  // Size of hyperparameters section
        uint32_t archSize;         // Size of architecture section
        uint32_t initSize;         // Size of initialization section
        uint32_t lrSchedulerSize;  // Size of learning rate scheduler section
    } __attribute__((packed));

    static void writeConfiguration(std::ofstream &file, const NetworkConfig &config)
    {
        // Write hyperparameters
        const auto &hyperparams = config.hyperparameters();
        ConfigHeader configHeader{};
        
        // Calculate section sizes
        size_t hyperparamsPos = static_cast<size_t>(file.tellp());
        file.write(reinterpret_cast<const char *>(&configHeader), sizeof(ConfigHeader));  // Placeholder

        // Write hyperparameters
        file.write(reinterpret_cast<const char *>(&hyperparams.learningRate), sizeof(double));
        file.write(reinterpret_cast<const char *>(&hyperparams.batchSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&hyperparams.dropout), sizeof(double));
        file.write(reinterpret_cast<const char *>(&hyperparams.epochs), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&hyperparams.samplesPerEpoch), sizeof(uint32_t));
        configHeader.hyperparamsSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - hyperparamsPos - sizeof(ConfigHeader));

        // Write architecture
        const auto &arch = config.architecture();
        size_t archPos = static_cast<size_t>(file.tellp());
        file.write(reinterpret_cast<const char *>(&arch.inputSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&arch.outputSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&arch.hiddenLayers), sizeof(uint32_t));
        uint32_t numSizes = static_cast<uint32_t>(arch.hiddenSizes.size());
        file.write(reinterpret_cast<const char *>(&numSizes), sizeof(uint32_t));
        for (const auto &size : arch.hiddenSizes) {
            file.write(reinterpret_cast<const char *>(&size), sizeof(uint32_t));
        }
        configHeader.archSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - archPos);

        // Write initialization
        const auto &init = config.initialization();
        size_t initPos = static_cast<size_t>(file.tellp());
        file.write(reinterpret_cast<const char *>(&init.weightInit), sizeof(WeightInit));
        file.write(reinterpret_cast<const char *>(&init.biasInit), sizeof(BiasInit));
        configHeader.initSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - initPos);

        // Write learning rate scheduler
        const auto &lrScheduler = config.lrScheduler();
        size_t lrPos = static_cast<size_t>(file.tellp());
        uint32_t typeLen = static_cast<uint32_t>(lrScheduler.type.length());
        file.write(reinterpret_cast<const char *>(&typeLen), sizeof(uint32_t));
        file.write(lrScheduler.type.c_str(), typeLen);
        file.write(reinterpret_cast<const char *>(&lrScheduler.decayRate), sizeof(double));
        file.write(reinterpret_cast<const char *>(&lrScheduler.decaySteps), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&lrScheduler.minLR), sizeof(double));
        configHeader.lrSchedulerSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - lrPos);

        // Go back and write the header with correct sizes
        auto currentPos = file.tellp();
        file.seekp(hyperparamsPos);
        file.write(reinterpret_cast<const char *>(&configHeader), sizeof(ConfigHeader));
        file.seekp(currentPos);
    }

    enum class LayerType : uint32_t {
        LINEAR = 1,
        RELU = 2,
        SOFTMAX = 3
    };

    struct LayerHeader {
        uint32_t type_raw;  // Store as raw uint32_t instead of enum
        uint32_t inputSize;
        uint32_t outputSize;
        uint32_t activation;

        LayerHeader(LayerType t, uint32_t in, uint32_t out, uint32_t act) 
            : type_raw(static_cast<uint32_t>(t)), inputSize(in), outputSize(out), activation(act) {}
    } __attribute__((packed));

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
        //layers.push_back(std::make_shared<nn::Softmax<double>>());

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
        LayerHeader header(
            LayerType::LINEAR,
            static_cast<uint32_t>(layer->_weights.tensor().shape()[0]),
            static_cast<uint32_t>(layer->_weights.tensor().shape()[1]),
            0
        );

        std::cout << "Writing LINEAR layer with type " << static_cast<uint32_t>(LayerType::LINEAR) << std::endl;
        std::cout << "Input size: " << header.inputSize << ", Output size: " << header.outputSize << std::endl;

        const auto &weights = layer->_weights.tensor().datas();
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
        file.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(double));

        const auto &biases = layer->_biases.tensor().datas();
        file.write(reinterpret_cast<const char *>(biases.data()), biases.size() * sizeof(double));
    }

    static void writeReluLayer(std::ofstream &file)
    {
        LayerHeader header(LayerType::RELU, 0, 0, 0);
        std::cout << "Writing RELU layer with type " << static_cast<uint32_t>(LayerType::RELU) << std::endl;
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }

    static void writeSoftmaxLayer(std::ofstream &file)
    {
        LayerHeader header(LayerType::SOFTMAX, 0, 0, 0);
        std::cout << "Writing SOFTMAX layer with type " << static_cast<uint32_t>(LayerType::SOFTMAX) << std::endl;
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
