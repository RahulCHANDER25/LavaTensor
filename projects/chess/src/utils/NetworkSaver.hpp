/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** NetworkSaver
*/

#pragma once

#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include "nn/Linear.hpp"
#include "nn/ReLU.hpp"
#include "nn/Sequential.hpp"
#include "nn/Softmax.hpp"
#include "utils/NetworkConfig.hpp"
#include "utils/NetworkLoader.hpp"

namespace lava {

class NetworkSaver {
    public:
    static void saveNetwork(const std::shared_ptr<nn::Sequential<double>> &network, const std::string &filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not create network file: " + filename);
        }

        Header header{};
        std::memcpy(header.magic, MAGIC, 4);
        header.version = VERSION;
        header.archHash = computeArchHash(network);
        header.numLayers = countLayers(network);
        std::memset(header.reserved, 0, sizeof(header.reserved));

        file.write(reinterpret_cast<const char *>(&header), sizeof(header));

        auto config = NetworkLoader::getLastLoadedConfig();
        ConfigHeader configHeader{};

        size_t hyperparamsPos = static_cast<size_t>(file.tellp());
        file.write(reinterpret_cast<const char *>(&configHeader), sizeof(ConfigHeader)); // Placeholder

        const auto &hyperparams = config.hyperparameters();
        file.write(reinterpret_cast<const char *>(&hyperparams.learningRate), sizeof(double));
        file.write(reinterpret_cast<const char *>(&hyperparams.batchSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&hyperparams.dropout), sizeof(double));
        file.write(reinterpret_cast<const char *>(&hyperparams.epochs), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&hyperparams.samplesPerEpoch), sizeof(uint32_t));
        configHeader.hyperparamsSize =
            static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - hyperparamsPos - sizeof(ConfigHeader));

        const auto &arch = config.architecture();
        size_t archPos = static_cast<size_t>(file.tellp());
        file.write(reinterpret_cast<const char *>(&arch.inputSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&arch.outputSize), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&arch.hiddenLayers), sizeof(uint32_t));
        auto numSizes = static_cast<uint32_t>(arch.hiddenSizes.size());
        file.write(reinterpret_cast<const char *>(&numSizes), sizeof(uint32_t));
        for (const auto &size : arch.hiddenSizes) {
            file.write(reinterpret_cast<const char *>(&size), sizeof(uint32_t));
        }
        configHeader.archSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - archPos);

        const auto &init = config.initialization();
        size_t initPos = static_cast<size_t>(file.tellp());
        file.write(reinterpret_cast<const char *>(&init.weightInit), sizeof(WeightInit));
        file.write(reinterpret_cast<const char *>(&init.biasInit), sizeof(BiasInit));
        configHeader.initSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - initPos);

        const auto &lrScheduler = config.lrScheduler();
        size_t lrPos = static_cast<size_t>(file.tellp());
        uint32_t typeLen = static_cast<uint32_t>(lrScheduler.type.length());
        file.write(reinterpret_cast<const char *>(&typeLen), sizeof(uint32_t));
        file.write(lrScheduler.type.c_str(), typeLen);
        file.write(reinterpret_cast<const char *>(&lrScheduler.decayRate), sizeof(double));
        file.write(reinterpret_cast<const char *>(&lrScheduler.decaySteps), sizeof(uint32_t));
        file.write(reinterpret_cast<const char *>(&lrScheduler.minLR), sizeof(double));
        configHeader.lrSchedulerSize = static_cast<uint32_t>(static_cast<size_t>(file.tellp()) - lrPos);

        auto currentPos = file.tellp();
        file.seekp(hyperparamsPos);
        file.write(reinterpret_cast<const char *>(&configHeader), sizeof(ConfigHeader));
        file.seekp(currentPos);

        for (const auto &layer : network->layers()) {
            if (auto linear = std::dynamic_pointer_cast<nn::Linear<double>>(layer)) {
                writeLinearLayer(file, linear);
            } else if (auto relu = std::dynamic_pointer_cast<nn::ReLU<double>>(layer)) {
                writeReluLayer(file);
            } else if (auto softmax = std::dynamic_pointer_cast<nn::Softmax<double>>(layer)) {
                writeSoftmaxLayer(file);
            }
        }

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
        uint32_t hyperparamsSize;
        uint32_t archSize;
        uint32_t initSize;
        uint32_t lrSchedulerSize;
    } __attribute__((packed));

    enum class LayerType : uint32_t {
        LINEAR = 1,
        RELU = 2,
        SOFTMAX = 3
    };

    struct LayerHeader {
        uint32_t type_raw; // Store as raw uint32_t instead of enum
        uint32_t inputSize;
        uint32_t outputSize;
        uint32_t activation;

        LayerHeader(LayerType t, uint32_t in, uint32_t out, uint32_t act)
            : type_raw(static_cast<uint32_t>(t)), inputSize(in), outputSize(out), activation(act)
        {
        }
    } __attribute__((packed));

    static uint64_t computeArchHash(const std::shared_ptr<nn::Sequential<double>> &network)
    {
        uint64_t hash = 0;
        for (const auto &layer : network->layers()) {
            if (auto linear = std::dynamic_pointer_cast<nn::Linear<double>>(layer)) {
                hash = hash * 31 + linear->_weights.tensor().shape()[0];
                hash = hash * 31 + linear->_weights.tensor().shape()[1];
            }
        }
        return hash;
    }

    static uint32_t countLayers(const std::shared_ptr<nn::Sequential<double>> &network)
    {
        return network->layers().size();
    }

    static void writeLinearLayer(std::ofstream &file, const std::shared_ptr<nn::Linear<double>> &layer)
    {
        LayerHeader header(
            LayerType::LINEAR,
            static_cast<uint32_t>(layer->_weights.tensor().shape()[0]),
            static_cast<uint32_t>(layer->_weights.tensor().shape()[1]),
            0
        );

        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
        const auto &weights = layer->_weights.tensor().datas();
        file.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(double));

        const auto &biases = layer->_biases.tensor().datas();
        file.write(reinterpret_cast<const char *>(biases.data()), biases.size() * sizeof(double));
    }

    static void writeReluLayer(std::ofstream &file)
    {
        LayerHeader header(LayerType::RELU, 0, 0, 0);
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }

    static void writeSoftmaxLayer(std::ofstream &file)
    {
        LayerHeader header(LayerType::SOFTMAX, 0, 0, 0);
        file.write(reinterpret_cast<const char *>(&header), sizeof(header));
    }
};

} // namespace lava
