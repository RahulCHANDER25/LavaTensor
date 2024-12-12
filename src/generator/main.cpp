/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** main
*/

#include <filesystem>
#include <iostream>
#include "ArgParser.hpp"
#include "generator/NetworkGenerator.hpp"
#include "utils/NetworkConfig.hpp"

int main(int argc, char *argv[])
{
    try {
        auto args = ArgParser::parseGeneratorArgs(argc, argv);

        for (const auto &[configFile, nbNetworks] : args.configs) {
            auto config = lava::NetworkConfig::fromFile(configFile);

            std::string baseName = std::filesystem::path(configFile).stem().string();
            for (int i = 1; i <= nbNetworks; i++) {
                std::string outputFile = baseName + "_" + std::to_string(i) + ".nn";
                std::cout << "Generating network " << i << " from " << configFile << " to " << outputFile << std::endl;
                lava::NetworkGenerator::generateNetwork(config, outputFile);
            }
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 84;
    }
}
