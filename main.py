import numpy as np
import matplotlib.pyplot as plt
import load_data as data
import neural_network as nn

# choose dataset
while True:
    print("Choose dataset (input: '1' or '2')")
    print("1. MNIST")
    print("2. CIFAR-10")
    dataset = input()
    if int(dataset) == 1:
        X_train, X_test, y_train, y_test = data.load_mnist_data()
        print("MNIST data loaded...")
        break
    elif int(dataset) == 2:
        X_train, X_test, y_train, y_test = data.load_cifar_data()
        print("CIFAR-10 data loaded...")
        break

# normalize dataset
X_train = X_train.astype(float)/255
X_test = X_test.astype(float)/255

# add column of ones to the dataset
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# choose hidden layer activation function
while True:
    print("Choose hidden layer activation function (input: '1', '2' or '3')")
    print("1. Softplus")
    print("2. Hyperbolic Tangent")
    print("3. Cosine")
    activation_function = input()
    if 1 <= int(activation_function) <= 3:
        break

# neural network parameters
minibatch_size = 200
hidden_units = 300
activation_function = int(activation_function)
iterations = 300
learning_rate = 0.5/minibatch_size
lamda = 0.03
tolerance = 1e-6

"""
# check gradients
for i in range(1, 3):
    NN = nn.NeuralNetwork(X_train, X_test, y_train, y_test, minibatch_size, hidden_units, activation_function, 
                          iterations, learning_rate, lamda, tolerance)

    gradEw, numericalGrad = NN.gradcheck(i)
    print("The difference estimate for gradient of w" + str(i) + " is : ", np.max(np.abs(gradEw - numericalGrad)))
"""
# train neural network
NN = nn.NeuralNetwork(X_train, X_test, y_train, y_test, minibatch_size, hidden_units, activation_function,
                      iterations, learning_rate, lamda, tolerance)

costs, accuracies = NN.train()

accuracy = accuracies[len(accuracies)-1]
error = round((1-accuracy)*100, 2)
print("Accuracy = ", accuracy)
print("Error = " + "%.2f" % error + "%")

plt.subplot(1, 2, 1)
plt.plot(np.squeeze(costs))
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.title("Learning rate=" + str(learning_rate) + "\nLambda=" + str(lamda))
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.ylabel('Accuracy')
plt.xlabel('Iterations')
plt.title("Error=" + "%.2f" % error + "%")
plt.show()
