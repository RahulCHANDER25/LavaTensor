/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** main
*/

#include "ArgParser.hpp"
#include "FileHandler.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
  try {
    auto args = ArgParser::parseGeneratorArgs(argc, argv);

    for (const auto &[configFile, nbNetworks] : args.configs) {
      std::string config = FileHandler::readFile(configFile);
      for (int i = 1; i <= nbNetworks; i++) {
        std::string outputFile =
            configFile.substr(0, configFile.find_last_of('.')) + "_" +
            std::to_string(i) + ".nn";

        // TODO: Implement neural network generation
        std::cout << "Generating network " << i << " from " << configFile
                  << " to " << outputFile << std::endl;
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 84;
  }
}