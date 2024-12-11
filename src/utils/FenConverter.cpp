/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** FenConverter
*/

#include "FenConverter.hpp"
#include <cctype>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

const std::map<std::string, double> FenConverter::OUT_RESULTS = {
    {"Checkmate", 1.0},
    {"Check", 2.0},
    {"Stalemate", 3.0},
    {"Nothing", 4.0},
};

std::string FenConverter::getFenBoard(const std::string &fen)
{
    return fen.substr(0, fen.find(' '));
}

double FenConverter::convertFenPlacement(char c)
{
    if (std::isupper(c)) {
        return (c - ('A')); // Because 0 means a space (if c = 'A')
    }
    if (std::islower(c)) {
        return (c - ('a')) * -1; // Because 0 means a space (if c = 'A')
    }
    return c;
}

std::vector<double> FenConverter::convertBoard(const std::string &fen) // TODO: One hot encoding
{
    std::vector<double> board;
    std::istringstream iss(getFenBoard(fen));

    std::string piece; 
    while (std::getline(iss, piece, '/')) {
        for (char c: piece) {
            if (isdigit(c)) {
                for (char k = '0'; k < c; k++) {
                    board.push_back(0);
                }
            } else {
                board.push_back(convertFenPlacement(c));
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
    for (const auto &[key, val]: OUT_RESULTS) {
        if (label.find(key) == 0) {
            return val;
        }
    }
    return -1;
}
