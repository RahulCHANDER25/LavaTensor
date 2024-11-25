/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** main
*/

#include <iostream>
#include "ArgParser.hpp"
#include "ChessboardParser.hpp"

int main(int argc, char *argv[])
{
    try {
        auto args = ArgParser::parseAnalyzerArgs(argc, argv);
        {
            // TODO: #5 Implement neural network loading
            std::cout << "Loading neural network from: " << args.loadFile << std::endl;
        }

        auto boards = ChessboardParser::parseChessboardFile(args.inputFile);

        if (args.isPredictMode) {
            std::cout << "Running in prediction mode" << std::endl;
            for (const auto &board : boards) {
                {
                    // TODO: #4 Implement prediction logic
                    std::cout << "Analyzing position: " << board.fen << std::endl;
                    std::cout << "Nothing" << std::endl;
                }
            }
        } else if (args.isTrainMode) {
            {
                // TODO: #3 Implement training logic
                std::cout << "Running in training mode" << std::endl;
                std::string saveFile = args.saveFile.empty() ? args.loadFile : args.saveFile;
                std::cout << "Saving trained network to: " << saveFile << std::endl;
            }
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 84;
    }
}
