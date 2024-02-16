This project consists of building and training a neural network to recogniye handwritten digits from the mnist dataset.

The repository contains:
-training data
    this is needed to train the model
-testing data
    this is needed to test the model
-two python scripts
    main.py contains the NeuralNetwork class containig all necessary methods to instantiate, train and test the neural network object
    trainandsavemodel.py contains all the code for calling the methods to actually train and save the model. It loads the training and testing data aswell as previously trained weights and biases
-the folder 'weights_and_biases'
    this folder stores the weights and biases of the model after being trained

How to use the code:
It is possible to use the code, that is train the network, adjust prameters such as learning rate, the number of hidden layers, or the number of neurons per layer. For that you only need to modify and/or run the 'trainandsavemodel.py' script. When instantiating the model you can choose to load in the weights and biases that have previously been trained or you can disregard these parameters to initialise the model with random weights and biases. Saving the model will overwrite the weights and biases stored previously.
