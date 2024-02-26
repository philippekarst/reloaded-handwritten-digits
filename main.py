import neural_network
import numpy as np
import pandas as pd
import os
import sys

#load the data
def load_mnist():
    df_train = pd.read_csv(r".\mnist_train.csv")
    df_test = pd.read_csv(r"C:.\mnist_test.csv")
    df_train = pd.DataFrame(df_train).to_numpy() #shape is (60000,785) 60000 pictures, 1 label + 28x28 pixels, every row is a label together with 784 pixels that make up the image
    df_test = pd.DataFrame(df_test).to_numpy() #shape is (10000,785) 10000 pictures, 1 label + 28x28 pixels
    return df_train, df_test

#preprocess data
def preprocess_data(df_train, df_test):
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
    return X_train_batches, Y_train_batches, X_test, Y_test

#load model
def load_model():
    directory = 'weights_and_biases/'
    weights = []
    biases = []
    for filename in os.listdir(directory):
        if filename.startswith('weights_') and filename.endswith('.csv'):
            weights.append(pd.read_csv(os.path.join(directory, filename), header = None).to_numpy())
        if filename.startswith('biases_') and filename.endswith('.csv'):
            biases.append(pd.read_csv(os.path.join(directory, filename), header = None).to_numpy())
    return weights, biases

#save model, i.e. save weights and biases
# Create the directory if it doesn't exist
def save_model():
    directory = 'weights_and_biases/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len(model.weights)):
        np.savetxt(f"{directory}weights_{i}.csv",model.weights[i],delimiter=",")
    for i in range(len(model.biases)):
        np.savetxt(f"{directory}biases_{i}.csv",model.biases[i],delimiter=",")

#load the own handwritten digits
def load_pics():
    pic = pd.read_csv(r".\pics.csv", header=None)
    pic = pd.DataFrame(pic).to_numpy().transpose()
    labels = pic[:,0]
    pics = pic[:,1:]/255
    return pics, labels

# Ask user if he wants to use the existing model or train a new model
def existing_or_new():
    a=True
    while a == True:
        user_input = input("Do you want to use an existing model? Enter 'yes' or 'no': ").lower()
        # Interpret the input as a boolean value
        if user_input == 'yes':
            user_boolean = True
            a = False
        elif user_input == 'no':
            user_boolean = False
            a = False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
    return user_boolean

#ask user if he wants to test the network or also train the network if he wants to train als the learning rate and the number of iterations
def test_or_train():
    a = True
    iterations = None
    learning_rate = None
    while a == True:
        user_input = input("Do you want to test or train the model? Enter 'test' or 'train'.").lower()
        a = False
        if user_input != 'test' and user_input != 'train':
            print("Invalid input. Please enter 'test' or 'train'.")
            a = True
        if user_input == 'train':
            while True:
                try:
                    iterations = input("How many iteration should be performed? (100-300 iterations are recommmended): ")
                    iterations = int(iterations)
                    break
                except:
                    print("Invalid input. Please enter an integer.")
            while True:
                try:
                    learning_rate = input("What should the learning rate be? (0.1 is recommmended): ")
                    learning_rate = float(learning_rate)
                    break
                except:
                    print("Invalid input. Please enter a float.")
    return user_input, iterations, learning_rate

#create nn
def create_nn(user_boolean):
    if user_boolean == True:
        weights, biases = load_model()
        model = neural_network.NeuralNetwork(784, [10], 10, weights, biases)
        print("The existing model has been loaded.")
    else:
        model = neural_network.NeuralNetwork(784, [10], 10)
        print("A new model has been created. It is recommended to train the network before testing.")
    return model

#train the model according to the useres preferences
def test_or_train_nn(X_train_batches, Y_train_batches, X_test, Y_test, user_input, iterations, learning_rate):
    if user_input == 'test':
        model.test_network(X_test, Y_test)
    if user_input == 'train':
        model.SGD(X_train_batches, Y_train_batches, learning_rate, iterations)
        save_model()
        print("The model has finished training and has been saved.")
        model.test_network(X_test, Y_test)
        
# Check if the dataset files exist in the working directory
if not (os.path.isfile("mnist_train.csv") and os.path.isfile("mnist_test.csv")):
    print("Error: MNIST dataset files are missing from the working directory.")
    print("Please download the dataset files from the following link and place them in the working directory:")
    print("https://www.kaggle.com/datasets/oddrationale/mnist-in-csv")
    sys.exit()
else:
    try:
        df_train, df_test = load_mnist()
    except Exception as e:
        print("An error occurred while loading the MNIST dataset:", e)
        sys.exit()


X_train_batches, Y_train_batches, X_test, Y_test = preprocess_data(df_train, df_test)

user_existing_or_new = existing_or_new()

model = create_nn(user_existing_or_new)

user_test_or_train, iterations, learing_rate = test_or_train()

test_or_train_nn(X_train_batches, Y_train_batches, X_test, Y_test, user_test_or_train, iterations, learing_rate)










#pic = pd.read_csv(r".\pics.csv", header=None)
#pic = pd.DataFrame(pic).to_numpy().transpose()
#labels = pic[:,0]
#pics = pic[:,1:]/255
