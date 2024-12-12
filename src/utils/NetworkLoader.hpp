/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** NetworkLoader
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

class NetworkLoader {
    public:
    static std::shared_ptr<nn::Sequential<double>> loadNetwork(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open network file: " + filename);
        }

        // Read and validate header
        Header header{};
        file.read(reinterpret_cast<char *>(&header), sizeof(header));
        validateHeader(header);

        // Read layers
        std::vector<std::shared_ptr<nn::Module<double>>> layers;
        for (uint32_t i = 0; i < header.numLayers; i++) {
            LayerHeader layerHeader{};
            file.read(reinterpret_cast<char *>(&layerHeader), sizeof(layerHeader));

            switch (layerHeader.type) {
                case LayerType::LINEAR:
                    layers.push_back(readLinearLayer(file, layerHeader));
                    break;
                case LayerType::RELU:
                    layers.push_back(std::make_shared<nn::ReLU<double>>());
                    break;
                case LayerType::SOFTMAX:
                    layers.push_back(std::make_shared<nn::Softmax<double>>());
                    break;
                default:
                    throw std::runtime_error("Unknown layer type in network file");
            }
        }

        return std::make_shared<nn::Sequential<double>>(layers);
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

    static void validateHeader(const Header &header)
    {
        if (std::memcmp(header.magic, MAGIC, 4) != 0) {
            throw std::runtime_error("Invalid network file: wrong magic number");
        }
        if (header.version != VERSION) {
            throw std::runtime_error("Unsupported network file version");
        }
        if (header.numLayers == 0) {
            throw std::runtime_error("Invalid network file: no layers");
        }
    }

    static std::shared_ptr<nn::Linear<double>> readLinearLayer(std::ifstream &file, const LayerHeader &header)
    {
        auto layer = std::make_shared<nn::Linear<double>>(header.inputSize, header.outputSize);

        auto &weights = layer->_weights.tensor().datas();
        file.read(reinterpret_cast<char *>(weights.data()), weights.size() * sizeof(double));

        auto &biases = layer->_biases.tensor().datas();
        file.read(reinterpret_cast<char *>(biases.data()), biases.size() * sizeof(double));

        return layer;
    }
};

} // namespace lava
