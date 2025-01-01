# Research Paper - Neural Network

# Abstract

In this project we were able to make a machine learning model that is able to determinate the state of a chess board (given in Fen Notation).

The whole using only Deep Learning methods/tools like Supervised Learning, Tensors, Autograd, Hyper-parameters Optimizations and Multi-threading.

# Introduction

My_Torch is an Epitech Project where we have to create a Neural Network from scratch and train it to analyze chessboards given in the Fen Notation. We had to be able to save the neural network and being able to load it for prediction purposes.

This paper show you all our Architecture, our Benchmarks and our Results.

# Architecture

First of all we knew that the most important thing was to create a Tensor Library to make all the linear algebra calculus easier.

1. **Tensor Library**

A tensor is a multidimensional array that permits all sort of sets of linear algebras operations like matrix multiplication, vector product or simple operations like subtraction and multiplication.

The name of our library was ***LavaTensor*** as a reference of *PyTorch* [1] library in Python, which we took great inspiration of. We built a ***TensorArray*** which contains all the Linear Algebras calculus and then we have the ***Tensor*** class which had all the logic of our main architecture logic: ***Autograd***

1. **Tensor Autograd**

Autograd is also known as “**reverse mode automatic differentiation”** is a method used in the gradient descent in machine learning. During a forward pass the Autograd creates a computation graph which will be used in the backward pass by going through this graph reversely and applying the Chain Rule [2].

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image.png)

This is an example of Autograd Computation graph, the backward pass will go through the graph from bottom to the top *Tensors*, also known as **leafs** (here *weight* and *bias*), and the “**Accumulate**” the gradient calculated before in their Tensors.

1. **Neural Network Architecture**

With the Tensor Library and Autograd, we finally did some tests of networks’ architecture. We implemented simple Linear (Dense) Layers, ReLU activation Functions and Softmax Layer.

As for the loss function we use *Cross Entropy Loss* [3] which offers better performances for classification and we used *Stocastic Gradient Descent (SGD)* [4] to optimize parameters after the backprogation.

Here is an example of a neural network architecture we had:

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image%201.png)

Note that with the implementation of Cross-Entropy loss that we did implement has a softmax in it. So for *LavaTensor* the Softmax layer is not needed.

# Benchmarks

During the benchmarks we used **Matplotlib** to make really useful graphs to visualize everything we needed to know for the training.

We used different types of layers as you can see below:

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image%202.png)

Here we tried differents density of layers with the number of parameters/neurons (weights/biases).

Below you can see all the different learning rates we tried:

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image%203.png)

As we can see as the lower the learning rate is the more accurate the model is, but the learning is slower.

During our trainings we modeled this heatmap to determinate which parameters correlate to each others, and so which parameters are more important to change to make a better model.

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image%204.png)

As you can see the most important information is that hidden layers are correlated to each others. We can also see that the *learning rate (LR)* an Hyper-parameter that does not correlate a lot to any other parameter. In bigger network it is usually better to have low learning rate there are a lots of techniques to twink this parameter, but in this use case we know that we could freely change this value (at a certain point of course).

As we did the benchmark, we were able to find a suitable architecture/Hyper-parameters for our model. Those are our results.

# **Results**

We have finally find our best architecture is:

| Parameter | Value |
| --- | --- |
| Input Size | 768 |
| Number of Hidden Layer | 2 |
| Layer 1 | 512 |
| Layer 2 | 256 |
| Output | 6 |
| Activation | ReLU |
| LR | 0.022215673448668344 |
| MIN_LR | 0.0009126742519640651 |
| Initial LR | 0.01 |
| Batch Size | 240 |
| Loss Function | Cross Entropy (With Softmax) |
| Optimizer | SGD |

Thanks to Autograd we were able to had multi-threading for training.

Here is a graph of the network architectures we tried.

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image%205.png)

The model we are talking about is the trial 238, that finally during the training had a success rate near to 90%.

As you can see, in comparison to lots of others models the 238 one perform better than the other.

![image.png](Research%20Paper%20-%20Neural%20Network%2015c79d2dcced8188b18dd65ef78990f6/image%206.png)

The architecture of this model pays a huge role on it. It is a little model with not too much hidden layers (2), which reduces over-fitting and we did early stops on training when we detect that the loss was stagnating.

All of those graphs shows us that trial 238 is the best model we could produce to resolve this dataset.

# Conclusion

We find this project really interesting because it made us able to understand more deeply neural networks and machine learning by doing it from scratch. By having this great architecture we were even able to have a deeper understanding in PyTorch Internals functioning with the implementation of Autograd. As well the scientific part of this project with doing benchmarks and comparison made us look like real scientist which was really good !

# **Methodology**

We used GitHub Project for the organization of tasks and as we said before we first tried to have a good architecture and great tools in order to scalable project when doing the real neural network, the classifier model for chess.

# **Literature Review**

During this research paper, we learned a lot of things from different people/groups and used their researches for ours, so we would to thank them !

A big thank you to the association https://www.effisciences.org/en and their workshops that made us understand more deeply PyTorch and the Large Language Models (LLM).

We also thanks the YouTube channels https://www.youtube.com/c/3blue1brown for his work in Deep Learning and Neural Network explaining, that inspired us a lot. We also thank https://www.youtube.com/@elliotwaite for his vulgarization of Autograd PyTorch that made us able to deploy this big Architecture in this project.

We thank Ilias Grosy and Jordan Bankole, who manage this project in Epitech Paris.

# **Bibliography**

[1] - https://pytorch.org/docs/stable/index.html

[2] - https://en.wikipedia.org/wiki/Chain_rule

[3] - https://en.innovatiana.com/post/cross-entropy-loss

[4] - https://optimization.cbe.cornell.edu/index.php?title=Stochastic_gradient_descent

[5] - https://matplotlib.org/