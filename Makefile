##
## EPITECH PROJECT, 2024
## LavaTensor
## File description:
## Makefile
##

CXX := g++ -std=c++20
CXXFLAGS := -Wall -Wextra -O3

SRC_DIR := src

TARGET := my_torch_generator
SRCS := $(addprefix $(SRC_DIR)/, 				\
				main.cpp						\
		)

OBJS := $(SRCS:%.cpp=%.o)

INCLUDES := -I$(SRC_DIR)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJS)

fclean: clean
	rm -rf $(TARGET)

re: fclean all

.PHONY: all clean fclean re
