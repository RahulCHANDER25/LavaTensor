#!/bin/bash

# Build the project
echo "Building project..."
make re

# Generate a neural network
echo -e "\nGenerating neural network..."
./my_torch_generator examples/best_network.conf 1

# Train the network
echo -e "\nTraining network..."
./my_torch_analyzer --train --save trained_network.nn best_network_1.nn examples/training_positions.txt

## Test the network
#echo -e "\nTesting network..."
#./my_torch_analyzer --predict trained_network.nn examples/test_positions.txt