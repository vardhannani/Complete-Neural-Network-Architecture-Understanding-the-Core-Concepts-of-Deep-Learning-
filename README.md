# 1. Introduction to Neural Networks
Neural networks are algorithms inspired by the human brain, used to recognize patterns and solve complex problems. They are the foundation of deep learning and can be used for tasks like image classification, natural language processing, and even self-driving cars.


# 2. Basic Structure of a Neural Network
A neural network consists of layers of neurons, which are connected by weights. The basic structure includes:

1. Input Layer: Takes input data.
2. Hidden Layers: Layers between the input and output where processing takes place.
3. Output Layer: Produces the result.
4. Each layer is made of neurons (or nodes) that perform mathematical computations. These neurons are connected via weights.

Formula for a Single Neuron:

$$ z = \sum_{i=1}^{n} w_i x_i + b $$

where:

1. x = Input
2. w = Weights
3. B = bias

# 3. Activation Functions
Activation functions determine whether a neuron should be activated or not, based on the input signal.

1. Sigmoid: Converts input to a value between 0 and 1.

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$


2. ReLU: Applies a thresholding step to the input.
ReLU

$$ \text{ReLU}(x) = \max(0, x) $$


3. Tanh: Converts input to a value between -1 and 1.

$$ \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} $$

 Where, 

 e = Euler's Number (2.7182818)
 
# 4. Forward Propagation
Forward propagation is the process of passing the input through the network to get the output. It involves the following steps:

1. Multiply inputs by weights.
2. Add the bias.
3. Apply the activation function.

$$ a = \sigma(w \cdot x + b) $$

# 5. Loss Function and Cost Function
The loss function measures the error in the prediction. The cost function is typically the average of the loss function across all data points.

For binary classification, the loss function could be the Binary Cross-Entropy:

$$ \text{Loss} = -\left(y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})\right) $$

# 6. Backpropagation and Gradient Descent
Backpropagation is the algorithm used to update the weights and biases by propagating the error backward from the output layer to the input layer. It minimizes the loss function using Gradient Descent.

Gradient descent is used to update the weights:

$$ \[ w = w - \eta \cdot \frac{\partial \text{Loss}}{\partial w} \] $$

# 7. Neural Network Training Process
The neural network is trained by iterating over the dataset, performing forward propagation, calculating the loss, performing backpropagation, and updating the weights. This process is repeated for many iterations (epochs) until the network converges to a minimum loss.

