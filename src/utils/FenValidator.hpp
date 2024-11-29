/*
** EPITECH PROJECT, 2024
** LavaTensor
** File description:
** FenValidator
*/

#pragma once

#include <optional>
#include <regex>
#include <sstream>
#include <utility>

class FenValidator {
    private:
    static constexpr std::string VALID_PIECES = "rnbqkpRNBQKP";

    struct ValidationResult {
        bool isValid;
        std::string error;

        ValidationResult(bool valid = true, std::string msg = "") : isValid(valid), error(std::move(msg)) {}

        operator bool() const
        {
            return isValid;
        }
    };

    static bool isValidPiece(char piece)
    {
        return VALID_PIECES.find(piece) != std::string::npos;
    }

    static ValidationResult validatePiecePlacement(const std::string &position)
    {
        std::istringstream iss(position);
        std::string rank;
        int rankCount = 0;

        while (std::getline(iss, rank, '/')) {
            rankCount++;
            int squareCount = 0;

            for (char c : rank) {
                if (std::isdigit(c)) {
                    int spaces = c - '0';
                    if (spaces <= 0 || spaces > 8) {
                        return {false, "Invalid number of empty squares: " + std::to_string(spaces)};
                    }
                    squareCount += spaces;
                } else if (isValidPiece(c)) {
                    squareCount++;
                } else {
                    return {false, "Invalid piece character: " + std::string(1, c)};
                }
            }

            if (squareCount != 8) {
                return {
                    false,
                    "Rank " + std::to_string(rankCount) + " has " + std::to_string(squareCount) +
                        " squares instead of 8"
                };
            }
        }

        if (rankCount != 8) {
            return {false, "Found " + std::to_string(rankCount) + " ranks instead of 8"};
        }

        return {true};
    }

    static ValidationResult validateActiveColor(const std::string &color)
    {
        if (color != "w" && color != "b") {
            return {false, "Active color must be 'w' or 'b', got: " + color};
        }
        return {true};
    }

    static ValidationResult validateCastlingRights(const std::string &castling)
    {
        if (castling == "-") {
            return {true};
        }

        std::string rights = castling;
        std::sort(rights.begin(), rights.end());
        if (std::unique(rights.begin(), rights.end()) != rights.end()) {
            return {false, "Duplicate castling rights in: " + castling};
        }

        for (char c : castling) {
            if (std::string("KQkq").find(c) == std::string::npos) {
                return {false, "Invalid castling right: " + std::string(1, c)};
            }
        }
        return {true};
    }

    static ValidationResult validateEnPassant(const std::string &enPassant)
    {
        if (enPassant == "-") {
            return {true};
        }

        if (!std::regex_match(enPassant, std::regex("[a-h][36]"))) {
            return {false, "Invalid en passant square: " + enPassant + " (must be '-' or a valid square a3-h3/a6-h6)"};
        }
        return {true};
    }

    static ValidationResult validateHalfmoveClock(const std::string &halfmove)
    {
        try {
            int moves = std::stoi(halfmove);
            if (moves < 0) {
                return {false, "Halfmove clock cannot be negative: " + halfmove};
            }
            return {true};
        } catch (...) {
            return {false, "Invalid halfmove clock (must be a number): " + halfmove};
        }
    }

    static ValidationResult validateFullmoveNumber(const std::string &fullmove)
    {
        try {
            int moves = std::stoi(fullmove);
            if (moves <= 0) {
                return {false, "Fullmove number must be positive: " + fullmove};
            }
            return {true};
        } catch (...) {
            return {false, "Invalid fullmove number (must be a number): " + fullmove};
        }
    }

    public:
    static std::optional<std::string> validateFEN(std::string fen)
    {
        while (!fen.empty() && std::isspace(fen.front())) {
            fen.erase(0, 1);
        }
        while (!fen.empty() && std::isspace(fen.back())) {
            fen.pop_back();
        }

        if (fen.empty()) {
            return "FEN string is empty";
        }

        std::vector<std::string> parts;
        std::istringstream iss(fen);
        std::string part;

        while (iss >> part) {
            if (!part.empty()) {
                parts.push_back(part);
            }
        }

        if (parts.size() != 6) {
            return "FEN must have exactly 6 parts, found " + std::to_string(parts.size()) +
                ". Required format: '<position> <active_color> <castling> <en_passant> <halfmove> <fullmove>'";
        }

        try {
            ValidationResult result;

            result = validatePiecePlacement(parts[0]);
            if (!result) {
                return "Invalid piece placement: " + result.error;
            }

            result = validateActiveColor(parts[1]);
            if (!result) {
                return "Invalid active color: " + result.error;
            }

            result = validateCastlingRights(parts[2]);
            if (!result) {
                return "Invalid castling rights: " + result.error;
            }

            result = validateEnPassant(parts[3]);
            if (!result) {
                return "Invalid en passant: " + result.error;
            }

            result = validateHalfmoveClock(parts[4]);
            if (!result) {
                return "Invalid halfmove clock: " + result.error;
            }

            result = validateFullmoveNumber(parts[5]);
            if (!result) {
                return "Invalid fullmove number: " + result.error;
            }

            std::string position = parts[0];
            int whiteKings = std::count(position.begin(), position.end(), 'K');
            int blackKings = std::count(position.begin(), position.end(), 'k');

            if (whiteKings != 1 || blackKings != 1) {
                return "Invalid number of kings (must be exactly one per side). "
                       "Found " +
                    std::to_string(whiteKings) + " white and " + std::to_string(blackKings) + " black kings";
            }

            return std::nullopt;
        } catch (const std::exception &e) {
            return std::string("Unexpected error while validating FEN: ") + e.what();
        } catch (...) {
            return "Unexpected error while validating FEN";
        }
    }

    // static bool isValidFEN(const std::string &fen)
    // {
    //     return !validateFEN(fen).has_value();
    // }
};
