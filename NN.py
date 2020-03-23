import numpy as np
import pandas as pd
from numpy import exp, array, random, dot

class NeuralNetwork:
    def __init__(self,N_inputs,L):
        self.weights=[0.1, 0.2, 0.3, 0.4, 0.5]
        self.weights2=np.random.rand(1)
        self.d_weights2=np.zeros(1)
        self.d_weights=[0, 0, 0, 0, 0]
        self.d_bias1=np.zeros(1)
        self.d_bias2=np.zeros(1)
        self.bias1=np.random.rand(1)
        self.bias2=np.random.rand(1)
        self.output=np.zeros(L)
        self.learning_rate=0.1

    def activation(self,x):
        #return 1/(1+exp(-x)) #Sigmoid
        return np.maximum(0.01*x,x)#Leaky ReLU

# The derivative of the activation function.
# This is the gradient of the activatiom curve.
# It indicates how confident we are about the existing weight.
    def activation_derivative(self,x):
        #print((exp(-x))/(pow((1+exp(-x)),2)))
        #return (exp(-x))/(pow((1+exp(-x)),2))#Sigmoid
        if x<=0:
            return 0.01
        else:
            return 1

    def feedforward(self,inputs):
        self.z1=np.dot(inputs,self.weights)+self.bias1
        self.layer1=self.activation(self.z1)
        self.z2=(self.layer1*self.weights2)+self.bias2
        self.output=self.activation(self.z2)
        return self.output

    def backprop(self,inputs,y):
        self.loss_derivative_wrt_y = 2*(y - self.output)
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        self.d_weights2=self.learning_rate*(self.layer1* (self.loss_derivative_wrt_y * self.activation_derivative(self.z2)))

        self.d_bias2=self.learning_rate*(self.loss_derivative_wrt_y*self.activation_derivative(self.z2))

        self.d_weights=self.learning_rate*(inputs*(((self.loss_derivative_wrt_y * self.activation_derivative(self.z2))*self.weights2)*self.activation_derivative(self.z1)))

        self.d_bias1=self.learning_rate*((self.loss_derivative_wrt_y*self.activation_derivative(self.z2)* self.weights2)*self.activation_derivative(self.z1))

        # update the weights with the derivative (slope) of the loss function
        self.weights=np.add(self.weights,self.d_weights)
        self.bias1+=self.d_bias1
        self.weights2+=self.d_weights2
        self.bias2+=self.d_bias2
# We train the neural network through a process of trial and error.
# Adjusting the synaptic weights each time.
    def train(self,training_set_inputs,training_set_outputs,number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            self.feedforward(training_set_inputs)
            self.backprop(training_set_inputs,training_set_outputs)

    def think(self, inputs):
        # Pass inputs through our neural network
        return self.feedforward(inputs)


#Intialise a single neuron neural network.
neural_network = NeuralNetwork(5,256)
print ("Input layer-weights before training: ")
print (neural_network.weights)
print ("Output layer-weights before training: ")
print (neural_network.weights2)
print ("Input layer bias before training: ")
print (neural_network.bias1)
print ("Output layer bias before training: ")
print (neural_network.bias2)
# Train the neural network using a training set.
# Do it 10,000 times and make small adjustments each time.
training_set_inputs = np.array([19000/32767, 100/108, 30/150, 150/9000, 76/300000])
training_set_outputs = [0]
neural_network.train(training_set_inputs, training_set_outputs, 50)

print ("New  input layer-weights after training: ")
print (neural_network.weights)
print ("New  output layer-weights after training: ")
print (neural_network.weights2)
print ("New  input layer bias after training: ")
print (neural_network.bias1)
print ("New  output layer bias after training: ")
print (neural_network.bias2)

teste=neural_network.think(training_set_inputs)
accuracy=0-teste