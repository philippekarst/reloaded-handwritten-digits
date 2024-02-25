import numpy as np
import pandas as pd

#all vectors are represented horizonally and matrices are multiplied to vectors from the right
#the number of samples in the training data of the mnist dataset of handwritten digits is 60000, I use this number to describe the shape of different arrays at times

class NeuralNetwork:
    def __init__(self, input_size : int, hidden_sizes : list, output_size : int, weights = None, biases = None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes #this is a list of integers that describe the abount of neurons per hidden layer
        self.output_size = output_size
        
        #this concatenates the sizes of all layers
        layer_sizes = [input_size]
        layer_sizes.extend(hidden_sizes)
        layer_sizes.append(output_size)

        #initialize weight matrices and bias vectors, if no weights or biases are given, then they are initialized randomly
        if weights == None:
            self.weights = [np.random.uniform(-0.5, 0.5, [layer_sizes[i], layer_sizes[i+1]]) for i in range(len(layer_sizes)-1)] #this produces a list of random weight matrices
        else:
            self.weights = weights
        if biases == None:
            self.biases = [np.zeros((1, layer_sizes[i+1])) for i in range(len(layer_sizes)-1)] #this produces a list of random bias vectors (the first layer, i.e. input layer doesnt have a bias)
        else:
            self.biases = biases

    def forward(self, inputs):
        activations = [inputs] #this will be a list of the activations of all layers. Every element of the list corresponds to a list of activation vectors of one layer for every digit of the data. The shape of each element of the list is (60000,# of neurons in that layer) 
        for i in range(len(self.weights)):
            z = np.dot(activations[-1],self.weights[i]) + self.biases[i] #we propagate from layer i to layer i+1
            activation = self.sigmoid(z) #the layer must undergo a non linear activation function to be able to approximate non linear functions
            activations.append(activation)
        return activations[-1] #returning the activations of the last layer i.e. the output of the nn. Its shape is (60000,10)
    
    def backward(self, inputs, targets):
        #we need the list of all activations before and after applying sigmoid therefore we perform forward propagation
        activations = [inputs]
        zs = [inputs] #list of activations before applying the activation function, i.e. sigmoid
        for i in range(len(self.weights)):
            activation = np.dot(activations[-1],self.weights[i]) + self.biases[i]
            zs.append(activation)
            activation = self.sigmoid(activation)
            activations.append(activation)

        #we need the targets to be horizontal vectors of length 10
        targets = self.one_hot(targets)#the shape op this is (60000,10)
        
        #initialize list of derivatives of weights, biases and ativations with respect to the cost function
        dW=[]
        dB=[]
        dA=[2*(activations[-1]-targets)]#sice the loss function of one sample is sum_i=0^10(a_i-y_i)^2, its derivative with respect to activation a_i is 2(a_i-y_i). For now the list contains one element its an array of the partial derivatives of the the cost with respect to the activation of each neuron in the output layer for evers sample in the dataset. The shape is (60000,10)

        for i in range(len(activations)-1, 0, -1):
            dA.append(np.dot(dA[-1]*self.sigmoid_derivative(zs[i]),self.weights[i-1].transpose())) #The chain rule lets us iterate backwards through all layers. The elements of dA have shapes (60000,len(outputlayer)), (60000,len(hiddenlayer)),... , (60000,len(inputlayer))
        dA.reverse() #we must reverse the list such that the differential of the activations of the last layer are also last in the list dA
        for i in range(len(activations)-1,0,-1):
            dW.append(1/(inputs.shape[0])*np.dot(activations[i-1].T,self.sigmoid_derivative(zs[i])*dA[i])) #in the same spirit we can iterate backwards to find the partial derivative of the cost wrt each weight of the nn. Since we want the mean error over all samples we must add them all up and devide by the sample size inputs.shape[0], in our case 60000
            dB.append(np.mean(dA[i]*self.sigmoid_derivative(zs[i]), axis =0)) #the same idea goes for the partial derivatives of the cost wrt all the biases of the network
        dW.reverse() #the lists must be reversed since back prop starts from behind
        dB.reverse()
        return dW, dB
    
    def update_parameters(self, inputs, targets, alpha):
        dW, dB = self.backward(inputs, targets) #back propagation calculates the gradient of the loss, which is a function of all weights and biases. Hence dW and dB can be thought of as being vectors pointing in the direction of greatest ascent.
        for i in range(len(self.weights)):
            self.weights[i] -= alpha*dW[i] #since we want to decrease loss, we must ajust the weights and biases in the opposite direction of the gradient. alpha controles how far we step in that direction
            self.biases[i] -= alpha*dB[i]

    def gradient_descenct(self, inputs, targets, alpha, iterations): #this function simply iterates the the process of stepping in the direction of greatest descent for the loss function.
        for i in range(iterations):
            self.update_parameters(inputs, targets, alpha)
            
    def SGD(self, input_batches, target_batches, alpha, iterations):
        for i in range(iterations):
            for j in range(len(input_batches)):
                self.gradient_descenct(input_batches[j], target_batches[j], alpha, 1)
                if i % 10 == 0 and j == 0:
                    print(f"iteration {i}:")
                    self.test_network(input_batches[j], target_batches[j])
        
    def sigmoid(self, x): #we use the sigmoid as activation function for all layers
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x): #this is the derivative of the sigmoid function
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def one_hot(self, targets): #this function one_hot encodes the values 0-10, e.g. the value 2 will be transformed into (0,0,1,0,0,0,0,0,0,0)
        one_hotY = np.zeros([targets.size,10]) #the one_hot method should be applicable to a batch of samples therefore we need the extra dimension. in our example one_hotY.shape = (60000,10)
        for j in range(targets.size):
            for i in range(10):
                if targets[j] == i:
                    one_hotY[j,i] = 1
        return one_hotY
    
    def get_prediction(self, inputs):
        A = self.forward(inputs)
        return np.argmax(A, axis=1) #to get a prediction we return the index of the largest value along axis = 1 (column) the shape is (60000,) because 'inputs' is a batch of 60000 inputs

    def test_network(self, inputs, targets):
        k=0
        predictions = self.get_prediction(inputs)
        for i in range(len(predictions)):
            if predictions[i] == targets[i]:
                k = k+1 #we count the number of correct predictions
        print(f'the accuracy is {k/len(targets)}')