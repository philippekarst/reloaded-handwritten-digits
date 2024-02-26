This project consists of building and training a neural network to recogniye handwritten digits from the mnist dataset.

The repository contains:

-training data
    this is needed to train the model
    
-testing data
    this is needed to test the model
    
-two python scripts
    neural_network.py contains the NeuralNetwork class containig all necessary methods to instantiate, train and test the neural network object
    main.py contains all the code for experimenting with the neural network, that is training, saving, testing, adjusting hyperparameters, etc..
    
-the folder 'weights_and_biases'
    this folder stores the weights and biases of the model after being trained

How to use the code:

First download the Mnist dataset from "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv" and save it in the working directory under "mnist_train.csv" and "mnist_test.csv".
Simply run main.py and follow the instructions in the terminal.
Adjusting number of neurons and number of hidden layers is not yet possible via user input. However, it can easily be done by modifying the function "create_nn" in main.py.
