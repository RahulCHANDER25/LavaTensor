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
SRC_DIR_UTILS := src/utils/
LIB_DIR_UTILS := lib/

INCLUDES := -I$(SRC_DIR_GEN) \
           -I$(SRC_DIR_ANA) \
           -I$(SRC_DIR_UTILS) \
           -I$(LIB_DIR_UTILS) \
           -Isrc/

SRCS_GEN := $(addsuffix .cpp,               \
            lib/Tensor/TensorArray          \
            lib/Tensor/Tensor               \
            $(addprefix $(SRC_DIR_GEN)/,    \
                main                        \
            ))

SRCS_ANA := $(addsuffix .cpp,               \
            lib/Tensor/TensorArray          \
            lib/Tensor/Tensor               \
            $(addprefix $(SRC_DIR_UTILS),   \
                FenConverter                \
                NetworkLoader               \
            )                               \
            $(addprefix $(SRC_DIR_ANA)/,    \
                main                        \
                $(addprefix training/,      \
                    chessTraining           \
                )                           \
            ))

OBJS_GEN := $(SRCS_GEN:%.cpp=%.o)
OBJS_ANA := $(SRCS_ANA:%.cpp=%.o)

all: my_torch_generator my_torch_analyzer

my_torch_generator: $(OBJS_GEN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS_GEN) $(LDLIBS)

my_torch_analyzer: $(OBJS_ANA)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJS_ANA) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf $(OBJS_GEN) $(OBJS_ANA)

fclean: clean
	rm -rf my_torch_generator my_torch_analyzer

re: fclean all

.PHONY: all clean fclean re