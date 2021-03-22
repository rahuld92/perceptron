import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


# Read data from csv file
def load_data():
    # get data as arrays
    loaded_data = pd.read_csv('data.csv', header=None)
    # convert arrays to matrix
    data_matrix = np.asmatrix(loaded_data, dtype='float64')
    return data_matrix


def generate_random_data():
    separable = False
    while not separable:
        samples = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=1,
                                      n_clusters_per_class=1, flip_y=-1)
        red = samples[0][samples[1] == 0]
        blue = samples[0][samples[1] == 1]
        separable = any([red[:, k].max() < blue[:, k].min() or red[:, k].min() > blue[:, k].max() for k in range(2)])
    # plt.plot(red[:, 0], red[:, 1], 'r.')
    # plt.plot(blue[:, 0], blue[:, 1], 'b.')

    # convert into numpy objects
    red = np.array(red)
    blue = np.array(blue)

    n, m = red.shape
    n2, m2 = blue.shape

    label_red = np.ones((n, 1))
    label_blue = -1 * np.ones((n2, 1))

    red = np.hstack((red, label_red))
    blue = np.hstack((blue, label_blue))
    random_data = np.concatenate((red, blue))

    return np.asmatrix(random_data, dtype='float64')


# plt.show()


def plot_data(_data, _weights):
    plt.scatter(np.array(_data[:50, 0]), np.array(_data[:50, 1]), marker='o', label='-1')
    plt.scatter(np.array(_data[50:, 0]), np.array(_data[50:, 1]), marker='+', label='1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()

    w = _weights.tolist()[0]

    x1 = [x1 for x1 in range(-3, 3)]
    x2 = [(w[0] + w[1] * i) / (-w[2]) for i in x1]
    plt.plot(x1, x2, 'r.-')

    # decsion = np.empty()

    # Here i am calculating slope and intercept with given three weights
    # for i in np.linspace(np.amin(_data[:, :1]), np.amax(_data[:, :1])):
    #     if (w[0] /w[1]) != 0:
    #         slope = -(w[0] / w[2]) / (w[0] / w[1])
    #         intercept = -w[0] / w[2]
    #         y = (slope * i) + intercept
    #         np.append(decsion, [i, y])
    #         # plt.plot(i, y, 'r.-')
    # print(decsion)
    plt.show()


def perceptron(_data):
    features = _data[:, 0:2]
    labels = _data[:, -1]
    learning_rate = 0.5

    print(labels)
    print(features)

    # Initialize 'w' with zeros
    w = np.zeros(shape=(1, features.shape[1] + 1))

    # Iterate 10 times and Update 'w' 
    # for i in range(10):
    for i in range(40):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())

            prediction = 1.0 if y > 0 else -1.0
            delta = learning_rate * (label.item(0, 0) - prediction)

            if delta:
                misclassified += 1
                w += (delta * x)
        print(f'Iteration [{i}] - Miscalculation Count - {misclassified}')
    return w


data = generate_random_data()
# data = load_data()
weights = perceptron(data)
plot_data(data, weights)
