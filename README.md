# Handwritten Digits Recognition Neural Network

## Overview
This project involves building and training a neural network to recognize handwritten digits using the MNIST dataset. The repository includes:

- **Training Data:** Required for training the model.
- **Testing Data:** Used to test the trained model.
- **requirement.txt**
- **Two Python Scripts:**
  - `neural_network.py`: Contains the `NeuralNetwork` class with methods for instantiating, training, and testing the neural network object.
  - `main.py`: Contains code for experimenting with the neural network, such as training, saving, testing, adjusting hyperparameters, etc.
- **`weights_and_biases` Folder:** Stores the weights and biases of the model after being trained.

## How to Use the Code
1. Download the MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).
2. Save the dataset in the working directory under "mnist_train.csv" and "mnist_test.csv".
3. Install the required dependencies using the following command:
    ```
    pip install -r requirements.txt
    ```
3. Run `main.py` and follow the instructions in the terminal.

**Note:** Adjusting the number of neurons and hidden layers is not yet possible via user input. However, it can be easily done by modifying the function `create_nn` in `main.py`.

Feel free to experiment, train, test, and adjust hyperparameters using the provided scripts. The trained model's weights and biases will be saved in the `weights_and_biases` folder.
