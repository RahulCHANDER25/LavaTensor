/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** FileHandler
*/

#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

class FileHandler {
    public:
    static std::string readFile(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    static std::vector<std::string> readLines(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file: " + filename);
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.find_first_not_of(" \t\r\n") != std::string::npos) {
                lines.push_back(line);
            }
        }
        return lines;
    }

    static void writeFile(const std::string &filename, const std::string &content)
    {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }
        file << content;
    }
};
