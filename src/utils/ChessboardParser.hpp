/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ChessboardParser
*/

#pragma once

#include "FenValidator.hpp"
#include "FileHandler.hpp"

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

        for (size_t lineNum = 0; lineNum < lines.size(); ++lineNum) {
            std::istringstream iss(lines[lineNum]);
            ChessboardData data;

            std::string component;
            for (size_t i = 0; i < 6 && iss >> component; ++i) {
                if (i > 0) {
                    data.fen += ' ';
                }
                data.fen += component;
            }

            std::string remaining;
            if (std::getline(iss >> std::ws, remaining)) {
                data.expectedOutput = remaining;
            }

            auto error = FenValidator::validateFEN(data.fen);
            if (error) {
                throw std::runtime_error(
                    "Invalid FEN notation at line " + std::to_string(lineNum + 1) + ": " + error.value() +
                    "\nComplete FEN: " + data.fen
                );
            }

            boards.push_back(data);
        }
        return boards;
    }
};
