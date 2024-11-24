##
## EPITECH PROJECT, 2024
## LavaTensor
## File description:
## Makefile
##

CXX := g++ -std=c++23
CXXFLAGS := -Wall -Wextra -O3

SRC_DIR_GEN := src/generator
SRC_DIR_ANA := src/analyzer

SRCS_GEN := $(addsuffix .cpp,               \
            $(addprefix $(SRC_DIR_GEN)/,    \
                main                         \
            ))

SRCS_ANA := $(addsuffix .cpp,               \
            $(addprefix $(SRC_DIR_ANA)/,    \
                main                         \
            ))

OBJS_GEN := $(SRCS_GEN:%.cpp=%.o)
OBJS_ANA := $(SRCS_ANA:%.cpp=%.o)

INCLUDES_GEN := -I$(SRC_DIR_GEN)
INCLUDES_ANA := -I$(SRC_DIR_ANA)

all: my_torch_generator my_torch_analyzer

my_torch_generator: $(OBJS_GEN)
	$(CXX) $(CXXFLAGS) $(INCLUDES_GEN) -o $@ $(OBJS_GEN) $(LDLIBS)

my_torch_analyzer: $(OBJS_ANA)
	$(CXX) $(CXXFLAGS) $(INCLUDES_ANA) -o $@ $(OBJS_ANA) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES_GEN) $(INCLUDES_ANA) -c $< -o $@

clean:
	rm -rf $(OBJS_GEN) $(OBJS_ANA)

fclean: clean
	rm -rf my_torch_generator my_torch_analyzer

re: fclean all

.PHONY: all clean fclean re