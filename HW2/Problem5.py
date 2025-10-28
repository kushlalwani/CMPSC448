import math
import random
import numpy as np

def bgd_l2(data, y, w, eta, delta, lam, num_iter):

    n, d = data.shape  # extract size of dataset and number of features
    new_w = np.copy(w)  # create new weight vector
    history_fw = []  # Store function values

    for _ in range(num_iter):
        gradient = np.zeros(d)  # Initialize gradient
        loss = 0  # Initialize loss function value

        for i in range(n):
            x_i = data[i]  # Feature vector
            y_i = y[i]  # Target value
            pred = np.dot(new_w, x_i)  # Compute prediction w * x

            # Follow function definition
            if y_i >= pred + delta:
                gradient += -2 * (y_i - pred - delta) * x_i #derivative of function with respect to x_i
                loss += (y_i - pred - delta) ** 2
            elif y_i <= pred - delta:
                gradient += -2 * (y_i - pred + delta) * x_i #derivative of function with respect to x_i
                loss += (y_i - pred + delta) ** 2

        # Add regularization term to gradient
        gradient = (gradient / n) + (2 * lam * new_w)

        # Update weights using gradient descent
        new_w = new_w - eta * gradient

        # Compute full objective function value and store
        reg_term = lam * np.sum(new_w ** 2)  # regularization term
        fw = (loss / n) + reg_term  # Compute full loss function
        history_fw.append(fw)

    return new_w, history_fw  # Return updated weights and loss history


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):

    n, d = data.shape
    new_w = np.copy(w)
    history_fw = []

    for t in range(1, num_iter + 1):  # Start iterations from 1
        # Select data point
        if i == -1:
            ind = np.random.randint(0, n)  # select random point
        else:
            ind = i  # Use the specific data point

        x_i = data[ind] #feature vector
        y_i = y[ind] #target vector
        pred = np.dot(new_w, x_i) # Compute prediction w * x

        # Compute learning rate
        learn_rate = eta / np.sqrt(t)

        # Follow function definition
        gradient = np.zeros(d)
        loss = 0

        if y_i >= pred + delta:
            gradient = -2 * (y_i - pred - delta) * x_i #derivative of function with respect to x_i
            loss = (y_i - pred - delta) ** 2
        elif y_i <= pred - delta:
            gradient = -2 * (y_i - pred + delta) * x_i #derivative of function with respect to x_i
            loss = (y_i - pred + delta) ** 2

        # Add regularization term
        gradient += 2 * lam * new_w

        # Update weights
        new_w = new_w - learn_rate * gradient

        # Compute objective function value
        reg_term = lam * np.sum(new_w ** 2)
        fw = loss + reg_term
        history_fw.append(fw)

        # dont iterate again if i != -1
        if i != -1:
            break

    return new_w, history_fw

