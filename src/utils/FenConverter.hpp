/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** FenConverter
*/

#pragma once

#include <map>
#include <string>
#include <vector>

class FenConverter {
public:
    static const std::map<std::string, double> OUT_RESULTS;

    FenConverter() = default;
    ~FenConverter() = default;

    static std::vector<double> convertBoard(const std::string &fen);

    static double convertBoardLabel(const std::string &label);

private:
    static std::string getFenBoard(const std::string &fen);

    static double convertFenPlacement(char c);
};
