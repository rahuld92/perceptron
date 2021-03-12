import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    _data = pd.read_csv('data.csv', header=None)
    _data = _data[:100]
    _data = np.asmatrix(_data, dtype='float64')
    return _data


def plot_data(_data):
    plt.scatter(np.array(_data[:50, 0]), np.array(_data[:50, 1]), marker='o', label='setosa')
    plt.scatter(np.array(_data[50:, 0]), np.array(_data[50:, 1]), marker='x', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend()
    plt.show()


def perceptron(_data, _num_iter):
    features = _data[:, 0:1]
    labels = _data[:, -1]

    # set weights to zero
    w = np.zeros(shape=(1, features.shape[1] + 1))

    __misclassified = []

    for epoch in range(_num_iter):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())
            target = 1.0 if (y > 0) else 0.0

            delta = (label.item(0, 0) - target)

            if delta:  # misclassified
                misclassified += 1
                w += (delta * x)

        __misclassified.append(misclassified)
    return w, __misclassified


data = load_data()
plot_data(data)
print(data)
num_iter = 10
w, misclassified_ = perceptron(data, num_iter)

epochs = np.arange(1, num_iter+1)
plt.plot(epochs, misclassified_)
plt.xlabel('iterations')
plt.ylabel('misclassified')
plt.show()
