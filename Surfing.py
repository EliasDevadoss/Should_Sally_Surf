#!/usr/bin/env python

import numpy as np
import pandas as pd
import random
import math
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


class Value:
    
    def __init__(self, data=None, label="", _parents=set(), _operation=""):
        """
        Constructor for a value node that holds data + gradient + how the value was produced
        """
        
        # initialize data randomly between -1 and 1 if no data was provided
        self.data = random.uniform(-1, 1) if data is None else data
        
        # initialize a gradient and label used for printing the value
        self.grad = 0
        self.label = label
        
        # initialize how the value was created (what operation using what inputs)
        # initialize how the backward pass for this operation executes
        self._parents = set(_parents)
        self._operation = _operation
        self._backward = lambda: None
    
    def __repr__(self):
        """
        Helper class used for printing the Value
        """

        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        """
        Implementing the + operator
        """
        
        # perform the operation
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _parents=(self, other), _operation='+')
    
        # assign how the gradients are propagated backwards
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward
    
        return out
    
    def __mul__(self, other):
        """
        Implementing the * operator
        """
        
        # perform the operation
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _parents=(self, other), _operation='*')
    
        # assign how the gradients are propagated backwards
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            
        out._backward = _backward
      
        return out

                
    def __pow__(self, other):
        """
        Implementation of the power operator **
        """
        
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, _parents=(self,), _operation="**" + str(other))
        
        def _backward():
            self.grad += (other * (self.data**(other-1))) * out.grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        """
        Implementation of ReLU activation function
        """
        
        out = Value(0 if self.data < 0 else self.data, _parents=(self,), _operation=("ReLU"))
        
        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    def backward(self):
        """
        Calls the _backward() method for each node in reverse topological order
        """

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)
    
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def exp(self):
        """
        Implementing exponent
        """

        x = self.data

        # to prevent overflow
        x = min(x, 50)
        out = Value(math.exp(x), _parents=(self,), _operation="exp")
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """
        Implementing log
        """
        
        x = self.data
        if x <= 0:
            raise ValueError("Negative")
        out = Value(math.log(x), _parents=(self,), _operation="log")
        
        def _backward():
            self.grad += (1.0 / x) * out.grad
        
        out._backward = _backward
        return out
    
    def __float__(self): return float(self.data)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1


def accuracy(Y, Yhat):
    """
    Function for computing accuracy
    
    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):
        if y == yhat: acc += 1
    return acc/len(Y) * 100


def sigmoid(value, scale=0.5):
    """
    Returns sig(value) using the scale parameter
    """

    scale_value = value * -1 * scale
    exp = scale_value.exp()
    denom = Value(1.0) + exp
    return Value(1.0) / denom

def negative_loglikelihood(y, pY1):
    """
    Return negative loglikelihood for a single example based on the value of Y and p(Y=1 | ...)
    """

    if y == 1:
        return - pY1.log()
    else:
        return - (Value(1.0) - pY1).log()

def elbow_method(X, max_k=10):
    inertias = [] # within-cluster sum of squares
    k_values = range(1, max_k)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

def cluster_centers():
    # Cluster data
    columns = ["Cluster", "MM", "DD", "hh", "mm", "WVHT", "DPD", "APD", "MWD", "WTMP"]
    data = [
        [1, 2.347203, 0.921098, 14.480022, 7.663291, 259.230920, 15.501489],
        [2, 6.183879, 1.034960, 7.703525, 5.950794, 275.113372, 18.503688],
        [3, 7.702992, 0.749206, 15.251048, 6.679906, 239.966435, 20.394126],
        [4, 11.344486, 0.848286, 13.782287, 8.000581, 261.649085, 16.389625]
    ]

    columns = ["Cluster", "MM", "WVHT", "DPD", "APD", "MWD", "WTMP"]

    # Format the data: first column as int, rest as floats with 3 decimal places
    formatted_data = [
        [f"{int(row[0])}"] + [f"{val:.3f}" for val in row[1:]]
        for row in data
    ]

    # Plot table
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    table = ax.table(cellText=formatted_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(0.5, 1.2)
    plt.title("Cluster Centroids Data", fontsize=14)
    plt.tight_layout()
    plt.show()

def dim_reduction(X, kmeans):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=kmeans.labels_, cmap='tab10', alpha=0.7)
    legend = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend)
    plt.title("KMeans Visualization")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

class Neuron:
    """
    Class that represents a single neuron in a neural network.
    A neuron first computes a linear combination of its inputs + an intercept,
    and then (potentially) passes it through a non-linear activation function
    """
  
    def __init__(self, n_inputs):
        """
        Constructor which initializes a parameter for each input, and one parameter for an intercept
        """
        self.theta = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.intercept = Value(random.uniform(-1, 1))

    def __call__(self, x, relu=False, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator ()
        """
        
        # produce linear combination of inputs + intercept
        out = sum([self.theta[i] * x[i] for i in range(len(self.theta))]) + self.intercept
        
        if train_mode and dropout_proba > 0:
            if random.random() < dropout_proba:
                out = Value(0.0)
            else:
                out = out * (1.0 / (1.0 - dropout_proba))
        
        # activate using ReLU based on boolean flag
        if relu:
            return out.relu()
        return sigmoid(out)
    
    
    def parameters(self):
        """
        Method that returns a list of all parameters of the neuron
        """

        return self.theta + [self.intercept]


class Layer:
    """
    Class for implementing a single layer of a neural network
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Constructor initializes the layer with neurons that ta
        """

        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
  
    def __call__(self, x, relu=True, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator ()
        """
        
        # produces a list of outputs for each neuron
        outputs = [n(x, relu, dropout_proba, train_mode=train_mode) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
  
    def parameters(self):
        """
        Method that returns a list of all parameters of the layer
        """

        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Class for implementing a multilayer perceptron
    """
  
    def __init__(self, n_features, layer_sizes, learning_rate=0.01, dropout_proba=0.1):
        """
        Constructor that initializes layers of appropriate width
        """

        layer_sizes = [n_features] + layer_sizes
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.dropout_proba = dropout_proba
        self.learning_rate = learning_rate
        # boolean flag that determines whether we are currently training or testing
        # helpful for controlling how dropout is used
        self.train_mode = True
  
    def __call__(self, x):
        """
        Impelementing the call () operator which simply calls each layer in the net
        sequentially using outputs of previous layers
        """
        
        # use ReLU activation for all layers except the last one
        out = x
        for layer in self.layers[0:len(self.layers)-1]:
            out = layer(out, relu=True, dropout_proba =self.dropout_proba, train_mode=self.train_mode)
        return self.layers[-1](out, relu=False)
  
    def parameters(self):
        """
        Method that returns a list of all parameters of the neural network
        """

        return [p for layer in self.layers for p in layer.parameters()]
    
    def _zero_grad(self):
        """
        Method that sets the gradients of all parameters to zero
        """

        for p in self.parameters():
            p.grad = 0
            
    def fit(self, Xmat_train, Y_train, Xmat_val=None, Y_val=None, max_epochs=100, batch_size= 500, verbose=False):
        """
        Fit parameters of the neural network to given data using batched SGD.
        Update weights after computing average loss for each batch
        SGD ends after reaching a maximum number of epochs.

        Can optionally take in validation inputs as well to test generalization error.
        """
        
        # to allow for early stopping
        static_epochs = 0
        best_acc = 0

        samples = len(Xmat_train)

        # initialize all parameters randomly
        for p in self.parameters():
            p.data = random.uniform(-1, 1)
            
        # iterate over epochs
        for e in range(max_epochs):
            self.train_mode = True

            # shuffle the data
            indices = np.arange(samples)
            np.random.shuffle(indices)

            Xmat_train = Xmat_train[indices]
            Y_train = Y_train[indices]

            for i in range(0, samples, batch_size):
                end = i + batch_size
                X_batch = Xmat_train[i:end]
                Y_batch = Y_train[i:end]
                self._zero_grad()
                batch_loss = None

                for xvec, y in zip(X_batch, Y_batch):
                    pY1 = self(xvec)

                    # clip data to prevent log(<=0)
                    epsilon = 1e-6
                    pY1.data = np.clip(pY1.data, epsilon, 1.0 - epsilon)

                    loss = negative_loglikelihood(y, pY1)
                    if batch_loss is None:
                        batch_loss = loss
                    else:
                        batch_loss = batch_loss + loss

                batch_loss = batch_loss * (1.0 / len(X_batch))
                
                batch_loss.backward()
                for p in self.parameters():
                    p.data -= self.learning_rate * p.grad

            # test for early stopping
            self.train_mode = False
            
            if Xmat_val is not None:
                val_acc = accuracy(Y_val, self.predict(Xmat_val))
                if verbose:
                    train_acc = accuracy(Y_train, self.predict(Xmat_train))
                    print(f"Epoch {e}: Training accuracy {train_acc:.0f}%, Validation accuracy {val_acc:.0f}%")
                
                # early stopping
                if val_acc > best_acc:
                    best_acc = val_acc
                    static_epochs = 0
                else:
                    static_epochs += 1

                # stop if no improvement
                if static_epochs >= 3:
                    if verbose:
                        print(f"Early stopping at epoch {e}")
                    break
            elif verbose:
                train_acc = accuracy(Y_train, self.predict(Xmat_train))
                print(f"Epoch {e}: Training accuracy {train_acc:.0f}%")

            self.train_mode = True
        
        # at the end of training set train mode to be False
        self.train_mode = False

    def predict(self, Xmat):
        """
        Predict method which returns a list of 0/1 labels for the given inputs
        """

        return [int(self(x).data > 0.5) for x in Xmat]


def analyze_data():
    """
    Function to analyze data using a neural network
    """

    # Read in data
    data = pd.read_csv("meteorological_data.csv")

    # Drop unwanted columns
    data_clean = data.drop(columns=["WDIR",	"WSPD", "GST", "PRES", "ATMP", "DEWP", "VIS", "TIDE"])
    # Ensures wave height, dominant and average period, and wave direction and temperature are present
    data_clean = data_clean[(data_clean["WVHT"] != 99) & (data_clean["DPD"] != 99) & (data_clean["APD"] != 99) & (data_clean["MWD"] != 999) & (data_clean["WTMP"] != 999)]

    # Ensure year is an integer column
    data_clean['YY'] = data_clean['YY'].astype(int)

    # Split data by year
    train = data_clean[data_clean["YY"].between(2019, 2022)].copy()
    val = data_clean[data_clean["YY"] == 2023].copy()
    test = data_clean[data_clean["YY"] == 2024].copy()

    # Drop year from features
    train_no_yy = train.drop(columns=["YY"]).to_numpy(dtype=np.float64)
    val_no_yy = val.drop(columns=["YY"]).to_numpy(dtype=np.float64)
    test_no_yy = test.drop(columns=["YY"]).to_numpy(dtype=np.float64)

    # Normalize the data
    mean = train_no_yy.mean(axis=0)
    std = train_no_yy.std(axis=0)
    Xmat_train = (train_no_yy - mean) / std
    Xmat_val = (val_no_yy - mean) / std
    Xmat_test = (test_no_yy - mean) / std

    # For deciding how many clusters to have (decided on k=4)
    #elbow_method(Xmat_train, max_k=10)

    # Clustering using k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(Xmat_train)

    # Dimensional reduction to visualize clusters
    #dim_reduction(Xmat_train, kmeans)
    #original_centroids = (kmeans.cluster_centers_ * std) + mean
    #centroids = pd.DataFrame(original_centroids, columns=['MM', 'DD', 'hh', 'mm', 'WVHT', 'DPD', 'APD', 'MWD', 'WTMP'])
    #print(centroids)

    # Table of data for cluster centroids
    #cluster_centers()

    # Assign cluster labels to all splits, cluster #3 (2 with zero indexing) is good conditions
    Y_train = (kmeans.predict(Xmat_train) == 2).astype(int)
    Y_val = (kmeans.predict(Xmat_val) == 2).astype(int)
    Y_test = (kmeans.predict(Xmat_test) == 2).astype(int)

    # Logistic Regression
    #log_reg = LogisticRegression(random_state=42)
    #log_reg.fit(Xmat_train, Y_train)
    #test_preds = log_reg.predict(Xmat_test)
    #print("Test Accuracy:", accuracy(Y_test, test_preds))

    # Neural Network
    n, d = Xmat_train.shape

    architectures = {
        "15, 1": [15, 1],
        "8, 8, 1": [8, 8, 1],
        "4, 4, 4, 4, 1": [4, 4, 4, 4, 1]
    }

    learning_rates = [0.1, 0.01, 0.001]

    best_accuracy = 0
    best_model = None

    for rate in learning_rates:
        for architect, layers in architectures.items():
            model = MLP(n_features=d, layer_sizes=layers, learning_rate=rate, dropout_proba=0.5)
            model.fit(Xmat_train, Y_train, Xmat_val, Y_val, max_epochs=50, verbose=True)
            this_accuracy = accuracy(Y_val, model.predict(Xmat_val))
            print(f"Architecture: {architect}, learning rate: {rate}, accuracy:{this_accuracy}")

            if this_accuracy > best_accuracy:
                best_accuracy = this_accuracy
                best_model = model

    return best_model, Xmat_test, Y_test


def main():
    """
    Testing
    """
    random.seed(42)
    model, X_test, Y_test = analyze_data()

    # Test final model
    #test_acc = accuracy(Y_test, model.predict(X_test))
    #print(f"Test accuracy {test_acc:.0f}%")

if __name__ == "__main__":
    main()
