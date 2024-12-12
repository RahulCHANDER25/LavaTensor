/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** NetworkConfig
*/

#pragma once

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lava {

enum class WeightInit {
    XAVIER,
    HE,
    UNIFORM
};

enum class BiasInit {
    ZEROS,
    UNIFORM
};

class NetworkConfig {
    public:
    struct Architecture {
        size_t inputSize{};
        size_t hiddenLayers{};
        std::vector<size_t> hiddenSizes;
        size_t outputSize{};
    };

    struct Hyperparameters {
        double learningRate{};
        size_t batchSize{};
        std::string activation;
        double dropout{};
    };

    struct Initialization {
        WeightInit weightInit;
        BiasInit biasInit;
    };

    static NetworkConfig fromFile(const std::string &filename)
    {
        NetworkConfig config;
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: " + filename);
        }

        std::string line;
        std::string currentSection;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (line[0] == '[') {
                currentSection = line.substr(1, line.find(']') - 1);
                continue;
            }

            auto pos = line.find('=');
            if (pos != std::string::npos) {
                std::string key = line.substr(0, pos);
                std::string value = line.substr(pos + 1);
                config._parseKeyValue(currentSection, key, value);
            }
        }

        config._validate();
        return config;
    }

    const Architecture &architecture() const
    {
        return _architecture;
    }

    const Hyperparameters &hyperparameters() const
    {
        return _hyperparameters;
    }

    const Initialization &initialization() const
    {
        return _initialization;
    }

    private:
    Architecture _architecture;
    Hyperparameters _hyperparameters;
    Initialization _initialization{};

    void _parseKeyValue(const std::string &section, const std::string &key, const std::string &value)
    {
        if (section == "architecture") {
            _parseArchitecture(key, value);
        } else if (section == "hyperparameters") {
            _parseHyperparameters(key, value);
        } else if (section == "initialization") {
            _parseInitialization(key, value);
        }
    }

    void _parseArchitecture(const std::string &key, const std::string &value)
    {
        if (key == "input_size") {
            _architecture.inputSize = std::stoul(value);
        } else if (key == "hidden_layers") {
            _architecture.hiddenLayers = std::stoul(value);
        } else if (key == "hidden_sizes") {
            std::stringstream ss(value);
            std::string size;
            while (std::getline(ss, size, ',')) {
                _architecture.hiddenSizes.push_back(std::stoul(size));
            }
        } else if (key == "output_size") {
            _architecture.outputSize = std::stoul(value);
        }
    }

    void _parseHyperparameters(const std::string &key, const std::string &value)
    {
        if (key == "learning_rate") {
            _hyperparameters.learningRate = std::stod(value);
        } else if (key == "batch_size") {
            _hyperparameters.batchSize = std::stoul(value);
        } else if (key == "activation") {
            _hyperparameters.activation = value;
        } else if (key == "dropout") {
            _hyperparameters.dropout = std::stod(value);
        }
    }

    void _parseInitialization(const std::string &key, const std::string &value)
    {
        if (key == "weight_init") {
            if (value == "xavier") {
                _initialization.weightInit = WeightInit::XAVIER;
            } else if (value == "he") {
                _initialization.weightInit = WeightInit::HE;
            } else if (value == "uniform") {
                _initialization.weightInit = WeightInit::UNIFORM;
            }
        } else if (key == "bias_init") {
            if (value == "zeros") {
                _initialization.biasInit = BiasInit::ZEROS;
            } else if (value == "uniform") {
                _initialization.biasInit = BiasInit::UNIFORM;
            }
        }
    }

    void _validate() const
    {
        if (_architecture.inputSize == 0) {
            throw std::runtime_error("Input size must be greater than 0");
        }
        if (_architecture.outputSize == 0) {
            throw std::runtime_error("Output size must be greater than 0");
        }
        if (_architecture.hiddenLayers != _architecture.hiddenSizes.size()) {
            throw std::runtime_error("Number of hidden layers does not match hidden sizes");
        }
        if (_hyperparameters.learningRate <= 0) {
            throw std::runtime_error("Learning rate must be greater than 0");
        }
        if (_hyperparameters.batchSize == 0) {
            throw std::runtime_error("Batch size must be greater than 0");
        }
        if (_hyperparameters.dropout < 0 || _hyperparameters.dropout >= 1) {
            throw std::runtime_error("Dropout must be between 0 and 1");
        }
    }
};

} // namespace lava
