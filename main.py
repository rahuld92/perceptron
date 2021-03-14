import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Read data from csv file
def load_data():
    # get data as arrays
    loaded_data = pd.read_csv('data.csv', header=None)
    # convert arrays to matrix
    data_matrix = np.asmatrix(loaded_data, dtype='float64')
    return data_matrix


def plot_data(_data, _weights):
    plt.scatter(np.array(_data[:50, 0]), np.array(_data[:50, 1]), marker='o', label='setosa')
    plt.scatter(np.array(_data[50:, 0]), np.array(_data[50:, 1]), marker='+', label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend()

    w = _weights.tolist()[0]

    # x1 = [x1 for x1 in range(4, 8)]
    # print(x1)
    # x2 = [(w[0] + w[1] * i) / (-w[2]) for i in x1]
    # print(x2)
    # plt.plot(x1, x2)

     # Here i am calculating slope and intercept with given three weights
    for i in np.linspace(np.amin(_data[:,:1]),np.amax(_data[:,:1])):
        slope = -(w[0]/w[2])/(w[0]/w[1])  
        intercept = -w[0]/w[2]

        #y =mx+c, m is slope and c is intercept
        y = (slope*i) + intercept
        plt.plot(i, y, 'ko')

    plt.show()


def perceptron(_data):
    features = _data[:, 0:2]
    labels = _data[:, -1]
    learning_rate = 0.5

    # Initialize 'w' with zeros
    w = np.zeros(shape=(1, features.shape[1] + 1))

    # Iterate 10 times and Update 'w' 
    for i in range(10):
        misclassified = 0
        for x, label in zip(features, labels):
            x = np.insert(x, 0, 1)
            y = np.dot(w, x.transpose())

            prediction = 1.0 if y > 0 else -1.0
            delta = learning_rate * (label.item(0, 0) - prediction)

            if delta:
                misclassified += 1
                w += (delta * x)
        # Just a Cool Output
        print(f'Iteration [{i}] - Miscalculation Count - {misclassified}')
    return w


data = load_data()
weights = perceptron(data)
plot_data(data, weights)
