import numpy as np
import scipy.special as ss


# output layer activation function
def softmax(x, ax=1):
    m = np.max(x, axis=ax, keepdims=True)
    p = np.exp(x-m)
    return p / np.sum(p, axis=ax, keepdims=True)


# hidden layer activation functions (i=1: softplus, i=2: tanh, i=3: cos)
def h(a, i):
    if i == 1:
        return np.logaddexp(0, a)  # np.log(1 + np.exp(a))
    elif i == 2:
        return np.tanh(a)
    else:
        return np.cos(a)


# derivatives of hidden layer activation functions
def dh(a, i):
    if i == 1:
        return ss.expit(a)  # 1 / (1+np.exp(-a))
    elif i == 2:
        return 1-np.square(np.tanh(a))
    else:
        return -np.sin(a)


# the neural network
class NeuralNetwork:
    def __init__(self, x_train, x_test, y_train, y_test, minibatch_size, m, ha, iterations, eta, lamda, tol):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.minibatch_size = minibatch_size
        self.m = m  # hidden units
        self.ha = ha  # activation function of hidden units
        self.iterations = iterations
        self.eta = eta  # learning rate
        self.lamda = lamda
        self.tol = tol  # tolerance

        self.n, self.d = x_train.shape  # number of inputs, attributes
        self.k = 10  # classes

        np.random.seed(0)
        self.w1 = np.random.randn(self.m, self.d)  # weights
        self.w2 = np.random.randn(self.k, self.m+1)  # weights
        self.z1 = None  # hidden layer outputs
        self.z2 = None  # softmax values

    def forward_propagation(self, x_train):
        self.z1 = h(np.dot(x_train, self.w1.T), self.ha)
        self.z1 = np.hstack((np.ones((self.z1.shape[0], 1)), self.z1))  # z0 = 1
        self.z2 = softmax(np.dot(self.z1, self.w2.T))

    def back_propagation(self, x_train, y_train):
        y = self.z1.dot(self.w2.T)
        max_error = np.max(y, axis=1)

        ew = np.sum(y_train * y) - np.sum(max_error) - \
            np.sum(np.log(np.sum(np.exp(y - np.array([max_error, ] * y.shape[1]).T), 1))) - \
            (0.5 * self.lamda) * (np.sum(np.square(self.w2)) + np.sum(np.square(self.w1)))

        gradew2 = (y_train - self.z2).T.dot(self.z1) - self.lamda * self.w2

        a = np.dot(y_train - self.z2, self.w2).T
        a = np.delete(a, 0, 0)
        gradew1 = ((dh(np.dot(x_train, self.w1.T), self.ha)).T * a).dot(x_train) - self.lamda * self.w1

        return ew, gradew1, gradew2

    def update_weights(self, gradew1, gradew2):
        self.w2 = self.w2 + self.eta*gradew2
        self.w1 = self.w1 + self.eta*gradew1

    def train(self):
        costs = []
        all_ew = []
        accuracies = []
        total_ewold = -np.inf

        print("Sum of costs of all mini-batches per iteration: ")

        for i in range(self.iterations):
            # shuffle indexes so that x_train indexes match y_train
            permutation = np.random.permutation(self.x_train.shape[0])
            self.x_train = self.x_train[permutation]
            self.y_train = self.y_train[permutation]

            for n in range(0, self.x_train.shape[0], self.minibatch_size):
                x_train_mini = self.x_train[n:n + self.minibatch_size]
                y_train_mini = self.y_train[n:n + self.minibatch_size]

                self.forward_propagation(x_train_mini)
                ew, gradew1, gradew2 = self.back_propagation(x_train_mini, y_train_mini)
                self.update_weights(gradew1, gradew2)
                all_ew.append(ew)

            total_ew = np.sum(all_ew)
            costs.append(total_ew)
            print(str(i + 1) + ".", total_ew)
            if np.abs(total_ew - total_ewold) < self.tol:
                break
            total_ewold = total_ew
            del all_ew[:]

            accuracies.append(self.accuracy())

        return costs, accuracies

    def gradcheck(self, i):
        epsilon = 1e-6
        _list = np.random.randint(self.x_train.shape[0], size=5)
        self.x_train = np.array(self.x_train[_list, :])
        self.y_train = np.array(self.y_train[_list, :])
        w = "w" + str(i)

        numerical_grad = np.zeros(getattr(self, w).shape)
        for k in range(numerical_grad.shape[0]):
            for d in range(numerical_grad.shape[1]):
                # add epsilon to the w[k,d]
                w_init = np.copy(getattr(self, w))
                w_tmp = np.copy(getattr(self, w))
                w_tmp[k, d] += epsilon
                setattr(self, w, w_tmp)
                self.forward_propagation(self.x_train)
                e_plus, gradew1, gradew2 = self.back_propagation(self.x_train, self.y_train)
                setattr(self, w, w_init)

                # subtract epsilon to the w[k,d]
                w_init = np.copy(getattr(self, w))
                w_tmp = np.copy(getattr(self, w))
                w_tmp[k, d] -= epsilon
                setattr(self, w, w_tmp)
                self.forward_propagation(self.x_train)
                e_minus, gradew1, gradew2 = self.back_propagation(self.x_train, self.y_train)
                setattr(self, w, w_init)

                numerical_grad[k, d] = (e_plus - e_minus) / (2 * epsilon)

        gradew = "gradew" + str(i)
        return vars()[gradew], numerical_grad

    def accuracy(self):
        tz1 = h(np.dot(self.x_test, self.w1.T), self.ha)
        tz1 = np.hstack((np.ones((tz1.shape[0], 1)), tz1))
        outputs = softmax(np.dot(tz1, self.w2.T))

        prediction = np.argmax(outputs, 1)
        return np.mean(prediction == np.argmax(self.y_test, 1))
