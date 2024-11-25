/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ChessboardParser
*/

#pragma once

#include "FileHandler.hpp"

#include <stdexcept>
#include <string>
#include <vector>

class ChessboardParser {
    public:
    struct ChessboardData {
        std::string fen;
        std::string expectedOutput;
    };

    static std::vector<ChessboardData> parseChessboardFile(const std::string &filename)
    {
        std::vector<ChessboardData> boards;
        auto lines = FileHandler::readLines(filename);

        for (const auto &line : lines) {
            ChessboardData data;
            std::istringstream iss(line);
            iss >> data.fen;
            if (!isValidFEN(data.fen)) {
                throw std::runtime_error("Invalid FEN notation in file");
            }

            std::string output;
            if (iss >> output) {
                data.expectedOutput = output;
            }

            boards.push_back(data);
        }
        return boards;
    }

    private:
    static bool isValidFEN(const std::string &fen)
    {
        // TODO: #1 Improve the checks
        if (fen.empty()) {
            return false;
        }

        int slashCount = 0;
        for (char c : fen) {
            if (c == '/') {
                slashCount++;
            }
        }
        return slashCount == 7;
    }
};
