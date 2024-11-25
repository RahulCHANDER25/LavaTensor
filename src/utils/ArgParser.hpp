/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** Argparser
*/

#pragma once

#include <stdexcept>
#include <string>
#include <vector>

class ArgParser {
    public:
    struct GeneratorArgs {
        std::vector<std::pair<std::string, int>> configs;
    };

    struct AnalyzerArgs {
        bool isPredictMode{};
        bool isTrainMode{};
        std::string loadFile;
        std::string inputFile;
        std::string saveFile;
    };

    static GeneratorArgs parseGeneratorArgs(int argc, char *argv[])
    {
        if (argc < 3 || (argc % 2) != 1) {
            throw std::runtime_error("Invalid number of arguments\nUSAGE: ./my_torch_generator "
                                     "config_file_1 nb_1 [config_file_2 nb_2...]");
        }

        GeneratorArgs args;
        for (int i = 1; i < argc; i += 2) {
            std::string configFile = argv[i];
            int nbNetworks = 0;
            try {
                nbNetworks = std::stoi(argv[i + 1]);
                if (nbNetworks <= 0) {
                    throw std::runtime_error("Number of networks must be positive");
                }
            } catch (const std::exception &) {
                throw std::runtime_error("Invalid number of networks");
            }
            args.configs.emplace_back(configFile, nbNetworks);
        }
        return args;
    }

    static AnalyzerArgs parseAnalyzerArgs(int argc, char *argv[])
    {
        if (argc < 4) {
            throw std::runtime_error("Invalid number of arguments\nUSAGE: ./my_torch_analyzer [--predict "
                                     "| --train [--save SAVEFILE]] LOADFILE FILE");
        }

        AnalyzerArgs args;
        args.isPredictMode = false;
        args.isTrainMode = false;
        args.saveFile = "";

        int i = 1;
        if (std::string(argv[i]) == "--predict") {
            args.isPredictMode = true;
            i++;
        } else if (std::string(argv[i]) == "--train") {
            args.isTrainMode = true;
            i++;
            if (i + 2 < argc && std::string(argv[i]) == "--save") {
                args.saveFile = argv[i + 1];
                i += 2;
            }
        } else {
            throw std::runtime_error("Must specify either --predict or --train mode");
        }

        if (i + 1 >= argc) {
            throw std::runtime_error("Missing LOADFILE or FILE argument");
        }

        args.loadFile = argv[i];
        args.inputFile = argv[i + 1];

        return args;
    }
};
