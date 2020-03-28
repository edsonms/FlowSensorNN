import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from numpy import exp, array, random, dot

class NeuralNetwork:
    def __init__(self,N_inputs,N_outputs):
        self.weights=np.random.rand(N_inputs)*np.sqrt(1/(N_inputs+N_outputs))
        self.weights2=np.random.rand(1)*np.sqrt(1/(N_inputs+N_outputs))
        self.d_weights2=np.zeros(1)
        self.d_weights=[0, 0, 0, 0, 0]
        self.bias1=np.zeros(1)
        self.bias2=np.zeros(1)
        self.output=np.zeros(1)
        self.learning_rate=0.1

    def activation(self,x):
        #return 1/(1+exp(-x)) #Sigmoid
        if x<=0:
            return 0.01*x
        else:
            return x#Leaky ReLU

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

    def feedforward(self,inputs, size):
        self.z1=np.zeros(size)
        self.layer1=np.zeros(size)
        self.z2=np.zeros(size)
        self.output=np.zeros(size)
        for iteration in range(size):
            self.z1[iteration]=np.dot(inputs[iteration],self.weights)+self.bias1
            self.layer1[iteration]=self.activation(self.z1[iteration])
            self.z2[iteration]=(self.layer1[iteration]*self.weights2)+self.bias2
            self.output[iteration]=self.activation(self.z2[iteration])
        
        return self.output

    def backprop(self,inputs,y,size):
        self.loss_derivative_wrt_y = 2*(y - self.output)
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #Ativação do Z2, como fazer para todo o batch?Fazer um for aqui e ir somando
        self.sum_d_weights2=self.learning_rate*(self.layer1* (self.loss_derivative_wrt_y * self.activation_derivative(self.z2)))
        self.d_weights2 = np.average(self.sum_d_weights2)
        
        self.sum_d_bias2=self.learning_rate*(self.loss_derivative_wrt_y*self.activation_derivative(self.z2))
        self.d_bias2 = np.average(self.sum_d_bias2)
        
        self.sum_d_weights=self.learning_rate*(inputs*(((self.loss_derivative_wrt_y * self.activation_derivative(self.z2))*self.weights2)*self.activation_derivative(self.z1)))
        self.d_weights=average(sum_d_weights)
        
        self.sum_d_bias1=self.learning_rate*((self.loss_derivative_wrt_y*self.activation_derivative(self.z2)* self.weights2)*self.activation_derivative(self.z1))
        self.d_bias1=self.average(sum_d_bias1)

        # update the weights with the derivative (slope) of the loss function
        self.weights=np.add(self.weights,self.d_weights)
        self.bias1+=self.d_bias1
        self.weights2+=self.d_weights2
        self.bias2+=self.d_bias2
# We train the neural network through a process of trial and error.
# Adjusting the synaptic weights each time.
    def train(self,training_set_inputs,training_set_outputs,epochs,n_batches,mini_batch):
        for training_iterations in range (epochs):
            for iteration in range(n_batches):
                self.set_inputs=training_set_inputs[iteration*mini_batch:((iteration+1)*(mini_batch)),0:5]
                self.set_outputs=training_set_outputs[iteration*mini_batch:((iteration+1)*(mini_batch))]
                self.feedforward(self.set_inputs,mini_batch)
                self.backprop(self.set_inputs,self.set_outputs,mini_batch)

    def think(self, inputs):
        # Pass inputs through our neural network
        return self.feedforward(inputs)


#Intialise a single neuron neural network.
neural_network = NeuralNetwork(5,1)
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
data=pd.read_csv("https://raw.githubusercontent.com/edsonms/Sensor_MATLAB/master/Analise%20Dados/dataset_training.CSV",sep=',',header=None, names=["pw_AC", "zcd", "beat_frequency", "desvio", "psd_desvio","y"])
training_set_inputs = data.iloc[:,0:5]
training_set_inputs = training_set_inputs.to_numpy()
training_set_outputs = data.y
training_set_outputs = training_set_outputs.to_numpy()
training_set_inputs,training_set_outputs = shuffle(training_set_inputs,training_set_outputs)
neural_network.train(training_set_inputs, training_set_outputs, 1000,6,32)

print ("New  input layer-weights after training: ")
print (neural_network.weights)
print ("New  output layer-weights after training: ")
print (neural_network.weights2)
print ("New  input layer bias after training: ")
print (neural_network.bias1)
print ("New  output layer bias after training: ")
print (neural_network.bias2)
data2=pd.read_csv("https://raw.githubusercontent.com/edsonms/Sensor_MATLAB/master/Analise%20Dados/dataset_testing.CSV",sep=',',header=None, names=["pw_AC", "zcd", "beat_frequency", "desvio", "psd_desvio","y"])
testing_set_inputs = data2.iloc[:,0:5]
testing_set_inputs = testing_set_inputs.to_numpy()
testing_set_outputs = data2.y
testing_set_outputs = testing_set_outputs.to_numpy()
teste=neural_network.think(testing_set_inputs[60])
accuracy=0-teste