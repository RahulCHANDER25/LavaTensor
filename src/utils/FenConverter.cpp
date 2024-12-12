/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** FenConverter
*/

#include "FenConverter.hpp"
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

const std::map<std::string, double> FenConverter::OUT_RESULTS = {
    {"Checkmate White", 0.0},
    {"Checkmate Black", 1.0},
    {"Check White", 2.0},
    {"Check Black", 3.0},
    {"Stalemate", 4.0},
    {"Nothing", 5.0},
};

// Piece encoding:
// White pieces: K=0, Q=1, R=2, B=3, N=4, P=5
// Black pieces: k=6, q=7, r=8, b=9, n=10, p=11
// Empty square: all zeros

std::string FenConverter::getFenBoard(const std::string &fen)
{
    return fen.substr(0, fen.find(' '));
}

int FenConverter::getPieceIndex(char c)
{
    switch (c) {
        case 'K':
            return 0;
        case 'Q':
            return 1;
        case 'R':
            return 2;
        case 'B':
            return 3;
        case 'N':
            return 4;
        case 'P':
            return 5;
        case 'k':
            return 6;
        case 'q':
            return 7;
        case 'r':
            return 8;
        case 'b':
            return 9;
        case 'n':
            return 10;
        case 'p':
            return 11;
        default:
            return -1;
    }
}

std::vector<double> FenConverter::convertBoard(const std::string &fen)
{
    std::vector<double> board(static_cast<size_t>(64 * 12), 0.0);
    std::istringstream iss(getFenBoard(fen));

    int square = 0;
    std::string row;
    while (std::getline(iss, row, '/')) {
        for (char c : row) {
            if (std::isdigit(c)) {
                square += c - '0';
            } else {
                int pieceIndex = getPieceIndex(c);
                if (pieceIndex >= 0) {
                    board[square * 12 + pieceIndex] = 1.0;
                }
                square++;
            }
        }
    }
    return board;
}

double FenConverter::convertBoardLabel(const std::string &label)
{
    if (label.empty()) {
        std::cerr << "Careful no label here !\n";
        return -1;
    }

    // For Stalemate and Nothing: ignore the color part
    if (label.find("Stalemate") != std::string::npos) {
        return OUT_RESULTS.at("Stalemate");
    }
    if (label.find("Nothing") != std::string::npos) {
        return OUT_RESULTS.at("Nothing");
    }

    // For Checkmate and Check: use the full label with color
    auto it = OUT_RESULTS.find(label);
    if (it != OUT_RESULTS.end()) {
        return it->second;
    }

    std::cerr << "Unknown label: " << label << "\n";
    return -1;
}
