def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def plot_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    plt.title("Sigmoid Activation Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y)
    plt.show()


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def plot_tanh():
    x = np.arange(-8, 8, 0.1)
    y = tanh(x)
    plt.plot(x, y)
    plt.show()


def relu(x):
    return np.maximum(0, x)


def plot_relu():
    x = np.arange(-5, 5, 0.1)
    y = relu(x)
    plt.title("ReLU Activation Function")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(x, y)
    plt.show()


plot_sigmoid()

# def lrelu(x, w, b, alpha=0.5):
#     return [w * i + b if w * i + b > 0 else alpha * (w * i + b) for i in x]

#
# def plot_leakyrelu():
#     x = np.arange(-7, 7, 0.1)
#     y = leakyrelu(x)
#     plt.plot(x, y)
#     plt.show()
#
#
# def act_plot(x, act_fun, w, b):
#     fig = plt.figure(figsize=(12, 4))
#
#     plt.subplot(121)
#     for i in range(len(w)):
#         plt.plot(x, act_fun(x, w[i], 0), label='w=' + str(w[i]) + '   b=0', alpha=0.8)
#     plt.legend()
#
#     plt.subplot(122)
#     for i in range(len(b)):
#         plt.plot(x, act_fun(x, 1, b[i]), label='w=1   b=' + str(b[i]), alpha=0.8)
#     plt.legend()
#
#     fig.suptitle(str(act_fun.__name__))
#     plt.show()
#
#
# x = np.linspace(-20, 20, 2000)
# # w = [1,2,3,4]
# # b = [1,2,3,4]
# w = [0.3, 0.4]
# b = [1, 4]

# act_plot(x, lrelu, w, b)


import matplotlib.pyplot as plt
import numpy as np

# Generate data points
x = np.arange(-5, 5, 0.1)

# Compute Leaky ReLU for different values of alpha
alpha = 0.3  # alpha is the slope of the negative part
y_leaky_relu = np.maximum(x, x * alpha)

# Plotting the Leaky ReLU graph
plt.plot(x, y_leaky_relu)
plt.title("Leaky ReLU Activation Function")
plt.xlabel("X")
plt.ylabel("Y")

# Show the plot
# plt.show()
