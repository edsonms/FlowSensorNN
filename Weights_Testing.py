import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats as st
from numpy import exp, array, random, dot

class NeuralNetwork:
    def __init__(self,N_inputs,N_outputs):
        self.n_features=N_inputs
        #self.weights = [0.62628646, 0.23252184, -0.03082221, 0.53694563, 0.1705619]
        #self.weights = [0.60571474, 0.01902601, -0.01948036, 0.72821424, 0.00214008]
        #self.weights = [0.56904482, 0.25006691, 0.04847503, 0.39396229, 0.38802502]
        #self.weights = [0.63913552, -0.20362535, -0.00184209, 1.30839642, 0.01922245]
        #self.weights = [0.90283097, 0.00272952, -0.09473628, 0.8740702, 0.19200741]
        #self.weights = [0.44967768, 0.11325128, 0.03009625, 0.68629725, 0.33280218]
        self.weights = [-1.42121751, 0.78049657, 0.13603078, -0.97191038, -0.55348018]
        
        #self.weights2 = 0.68615644
        #self.weights2 = 0.73922057
        #self.weights2 = 0.66999889
        #self.weights2 = 0.63649289
        #self.weights2 = 0.60536165
        #self.weights2 = 0.81266283
        self.weights2 = -0.70397624
        
        #self.bias1 = -0.09480597
        #self.bias1 = -0.15470458
        #self.bias1 = -0.07004388
        #self.bias1 = -0.37036459
        #self.bias1 = -0.16001642
        #self.bias1 = -0.0871177
        self.bias1 = 1.49408153
        
        #self.bias2 = -0.01356739
        #self.bias2 = -0.00677469
        #self.bias2 = -0.1918039
        #self.bias2 = 0.04686167
        #self.bias2 = 0.02547376
        #self.bias2 = -0.02098692
        self.bias2 = 0.83152831
        self.output=np.zeros(1)

    def activation(self,x):
        #return 1/(1+exp(-x)) #Sigmoid
        if x<=0:
            return 0.01*x
        else:
            return x#Leaky ReLU

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

    def think(self, inputs,size):
        # Pass inputs through our neural network
        return self.feedforward(inputs,size)


#Intialise a single neuron neural network.
n_features = 5
data=pd.read_csv("https://raw.githubusercontent.com/edsonms/Sensor_MATLAB/master/Analise%20Dados/Dataset_Full.CSV",sep=',',header=None, names=["pw_AC", "zcd", "beat_frequency", "desvio", "psd_desvio","y"])
#data=data.drop(["beat_frequency","zcd"],axis=1)
testing_set_inputs = data.iloc[:,0:n_features]
testing_set_inputs = testing_set_inputs.to_numpy()
testing_set_outputs = data.y
testing_set_outputs = testing_set_outputs.to_numpy()

neural_network = NeuralNetwork(n_features,1)
teste=neural_network.think(testing_set_inputs,testing_set_inputs.shape[0])
prediction_error=testing_set_outputs-teste
mean = np.mean(np.abs(prediction_error))
median = np.median(np.abs(prediction_error))

print ("Prediction error mean: ")
print(mean)
print ("Prediction error median: ")
print(median)
print ("Prediction error max: ")
print(np.max(prediction_error))
