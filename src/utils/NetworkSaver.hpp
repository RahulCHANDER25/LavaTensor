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
};

} // namespace lava
