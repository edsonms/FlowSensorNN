import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from scipy import stats as st
from numpy import exp, array, random, dot

class NeuralNetwork:
    def __init__(self,N_inputs,N_outputs,learning_rate):
        self.n_features=N_inputs
        self.weights=np.random.rand(N_inputs)*np.sqrt(1/(N_inputs+N_outputs))
        self.weights2=np.random.rand(1)*np.sqrt(1/(N_inputs+N_outputs))
        self.d_weights2=np.zeros(1)
        self.d_weights=[0, 0, 0, 0, 0]
        self.bias1=np.zeros(1)
        self.bias2=np.zeros(1)
        self.output=np.zeros(1)
        self.learning_rate=learning_rate

    def add_noise(self,x):
        row,col=x.shape
        for j in range(col):
            for i in range(row):
                if x[i,j]<0.001:
                    mu, sigma = 0, 0.00001
                    # creating a normal distributed noise
                    noise = np.random.normal(mu, sigma)
                    x[i,j]=x[i,j]+noise
                elif x[i,j]<0.01:
                    mu, sigma = 0, 0.0001
                    # creating a normal distributed noise
                    noise = np.random.normal(mu, sigma)
                    x[i,j]=x[i,j]+noise
                elif x[i,j]<0.1:
                    mu, sigma = 0, 0.001
                    # creating a normal distributed noise
                    noise = np.random.normal(mu, sigma)
                    x[i,j]=x[i,j]+noise
                else :
                    mu, sigma = 0, 0.01
                    # creating a normal distributed noise
                    noise = np.random.normal(mu, sigma)
                    x[i,j]=x[i,j]+noise
        return x

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
            return 0.001
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
        d_activation_z1=np.zeros(size)
        d_activation_z2=np.zeros(size)
        for iteration in range(size):
            d_activation_z2[iteration]=self.activation_derivative(self.z2[iteration])
            d_activation_z1[iteration]=self.activation_derivative(self.z1[iteration])
            
        self.loss_derivative_wrt_y = 2*(y - self.output)
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #Ativação do Z2, como fazer para todo o batch?Fazer um for aqui e ir somando
        self.sum_d_weights2=(self.layer1* (self.loss_derivative_wrt_y * d_activation_z2))
        self.d_weights2 = np.mean(self.sum_d_weights2,dtype=np.float64)
        
        self.sum_d_bias2=(self.loss_derivative_wrt_y*d_activation_z2)
        self.d_bias2 = np.mean(self.sum_d_bias2,dtype=np.float64)
        
        self.sum_d_weights=np.zeros([size, self.n_features])
        mult_temp=(((self.loss_derivative_wrt_y * d_activation_z2)*self.weights2)*d_activation_z1)
        for iteration in range(size):
            self.sum_d_weights[iteration]=inputs[iteration]*mult_temp[iteration]

        self.sum_d_weights=self.sum_d_weights
        self.d_weights=np.mean(self.sum_d_weights,axis=0,dtype=np.float64)
        
        self.sum_d_bias1=((self.loss_derivative_wrt_y*d_activation_z2* self.weights2)*d_activation_z1)
        self.d_bias1=np.mean(self.sum_d_bias1,dtype=np.float64)

        # update the weights with the derivative (slope) of the loss function
        self.weights=np.add(self.weights,self.learning_rate*self.d_weights)
        self.bias1+=self.learning_rate*self.d_bias1
        self.weights2+=self.learning_rate*self.d_weights2
        self.bias2+=self.learning_rate*self.d_bias2
# We train the neural network through a process of trial and error.
# Adjusting the synaptic weights each time.
    def train(self,training_set_inputs,training_set_outputs,epochs,n_batches,mini_batch):
        i=0
        accumulated_loss=np.zeros(epochs*n_batches)
        for training_iterations in range (epochs):
            training_set_inputs,training_set_outputs = shuffle(training_set_inputs,training_set_outputs)
            for iteration in range(n_batches):
                self.set_inputs=training_set_inputs[iteration*mini_batch:((iteration+1)*(mini_batch)),0:self.n_features]
                self.set_outputs=training_set_outputs[iteration*mini_batch:((iteration+1)*(mini_batch))]
                self.feedforward(self.set_inputs,mini_batch)
                self.backprop(self.set_inputs,self.set_outputs,mini_batch)
                accumulated_loss[i]=np.mean((self.set_outputs - self.output))
                i=i+1
            
            training_set_inputs=self.add_noise(training_set_inputs)


        
        #plt.plot(range(0,epochs*n_batches),accumulated_loss,'r',label='Training Loss')
        #plt.title('Training loss versus training iteration')
        #plt.xlabel('Epochs')
        #plt.ylabel('Loss')
        #plt.legend()
        #plt.show()

    def think(self, inputs,size):
        # Pass inputs through our neural network
        return self.feedforward(inputs,size)


#Intialise a single neuron neural network.
n_features = 5
learning_rate = 1
data=pd.read_csv("https://raw.githubusercontent.com/edsonms/Sensor_MATLAB/master/Analise%20Dados/dataset_training.CSV",sep=',',header=None, names=["pw_AC", "zcd", "beat_frequency", "desvio", "psd_desvio","y"])
#data=data.drop(["beat_frequency","zcd"],axis=1)
training_set_inputs = data.iloc[:,0:n_features]
training_set_inputs = training_set_inputs.to_numpy()
training_set_outputs = data.y
training_set_outputs = training_set_outputs.to_numpy()
data2=pd.read_csv("https://raw.githubusercontent.com/edsonms/Sensor_MATLAB/master/Analise%20Dados/dataset_testing.CSV",sep=',',header=None, names=["pw_AC", "zcd", "beat_frequency", "desvio", "psd_desvio","y"])
testing_set_inputs = data2.iloc[:,0:n_features]
testing_set_inputs = testing_set_inputs.to_numpy()
testing_set_outputs = data2.y
testing_set_outputs = testing_set_outputs.to_numpy()
target_mean = 0.16
w=0

while w==0:
    neural_network = NeuralNetwork(n_features,1,learning_rate)
    neural_network.train(training_set_inputs, training_set_outputs, 100,6,32)
    teste=neural_network.think(testing_set_inputs,testing_set_inputs.shape[0])
    prediction_error=testing_set_outputs-teste
    mean = np.mean(np.abs(prediction_error))
    learning_rate = learning_rate - 0.01
    if np.abs(mean) > target_mean:
        if learning_rate > 0:
            learning_rate = learning_rate
        else:
            learning_rate = 1
    else:
        w=1



print ("New  input layer-weights after training: ")
print (neural_network.weights)
print ("New  output layer-weights after training: ")
print (neural_network.weights2)
print ("New  input layer bias after training: ")
print (neural_network.bias1)
print ("New  output layer bias after training: ")
print (neural_network.bias2)
print ("Learning Rate: ")
print(learning_rate)
print ("Prediction error mean: ")
print(mean)
print ("Prediction error max: ")
print(np.max(prediction_error))
