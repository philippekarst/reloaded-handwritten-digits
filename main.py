import neural_network
import numpy as np
import pandas as pd
import os

#load the data
df_train = pd.read_csv(r".\mnist_train.csv")
df_test = pd.read_csv(r"C:.\mnist_test.csv")
df_train = pd.DataFrame(df_train).to_numpy() #shape is (60000,785) 60000 pictures, 1 label + 28x28 pixels, every row is a label together with 784 pixels that make up the image
df_test = pd.DataFrame(df_test).to_numpy() #shape is (10000,785) 10000 pictures, 1 label + 28x28 pixels


#preprocess data, separate image from label, shuffle the data
df_train = df_train[np.random.permutation(df_train.shape[0]), :] #shuffle rows of the training data
X_train = df_train[:,1:]/255 #we remove the first column, i.e. the labels and devide by 255 to get floats between 0 and 1
Y_train = df_train[:,0] #we remove the pixels and get a row vector of all the labels
X_test = df_test[:,1:]/255 #we remove the first column, i.e. the labels and devide by 255 to get floats between 0 and 1
Y_test = df_test[:,0]
#create batches for SGD
X_train_batches = []
Y_train_batches = []
for i in range(60):
    X_train_batches.append(X_train[1000*i:1000*i+1000,:])
    Y_train_batches.append(Y_train[1000*i:1000*i+1000])



#load model
directory = 'weights_and_biases/'
weights = []
biases = []
for filename in os.listdir(directory):
    if filename.startswith('weights_') and filename.endswith('.csv'):
        weights.append(pd.read_csv(os.path.join(directory, filename), header = None).to_numpy())
    if filename.startswith('biases_') and filename.endswith('.csv'):
        biases.append(pd.read_csv(os.path.join(directory, filename), header = None).to_numpy())

#create neural network
model = neural_network.NeuralNetwork(784, [10], 10, weights, biases)

#train neural network
#model.gradient_descenct(X_train,Y_train, 0.1, 100)
model.SGD(X_train_batches,Y_train_batches, 0.1, 100)


#save model, i.e. save weights and biases
# Create the directory if it doesn't exist
directory = 'weights_and_biases/'
if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(len(model.weights)):
    np.savetxt(f"{directory}weights_{i}.csv",model.weights[i],delimiter=",")
for i in range(len(model.biases)):
    np.savetxt(f"{directory}biases_{i}.csv",model.biases[i],delimiter=",")


pic = pd.read_csv(r".\pics.csv", header=None)
pic = pd.DataFrame(pic).to_numpy().transpose()
labels = pic[:,0]
pics = pic[:,1:]/255


#test network on test dataset
model.test_network(pics,labels)