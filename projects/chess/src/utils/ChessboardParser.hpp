/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** ChessboardParser
*/

#pragma once

#include "FenConverter.hpp"
#include "FenValidator.hpp"
#include "FileHandler.hpp"

#include <string>
#include <vector>

class ChessboardParser {
    public:
    struct ChessboardData {
        std::string fen;
        std::vector<double> boardData;

        std::string expectedOutput;
        double outLabel = 0.f;
    };

    static std::vector<ChessboardData> parseChessboardFile(const std::string &filename)
    {
        std::vector<ChessboardData> boards;
        auto lines = FileHandler::readLines(filename);

        for (size_t lineNum = 0; lineNum < lines.size(); ++lineNum) {
            if (lines[lineNum].empty() || lines[lineNum][0] == '#') {
                continue;
            }

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

            data.boardData = FenConverter::convertBoard(data.fen);
            if (!data.expectedOutput.empty()) {
                data.outLabel = FenConverter::convertBoardLabel(data.expectedOutput);
            }
            boards.push_back(data);
        }
        return boards;
    }
};
