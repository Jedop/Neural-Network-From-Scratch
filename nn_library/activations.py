try:
    import cupy as np
    print("GPU (CuPy) detected. Running on GPU.")
except ImportError:
    import numpy as np
    print("GPU (CuPy) not found. Running on CPU with NumPy.")

def softmax(Z, derivative : bool = 0):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)  # subtract column max
    exp_Z = np.exp(Z_shift)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True) if not derivative else 1

def ReLU(Z, derivative : bool = 0):
    return np.maximum(0, Z) if not derivative else Z > 0

def sigmoid(Z, derivative : bool = 0):
    return 1/(1+np.exp(-Z)) if not derivative else np.exp(-Z)*sigmoid(Z)**2

def tanh(Z, derivative : bool = 0):
    return 2*sigmoid(2*Z) - 1 if not derivative else 4*sigmoid(2*Z, 1)

def LeakyReLU(Z, a=0.01, derivative : bool = 0):
    return np.where(Z > 0, Z, a * Z) if not derivative else np.where(Z > 0, 1, a)

def one_hot(Y):
    Y = np.array(Y).astype(int)
    num_classes = int(Y.max().item()) + 1  # Convert CuPy scalar to Python int
    one_hot_Y = np.zeros((Y.size, num_classes))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

activation_functions = {
    "relu" : ReLU,
    "leakyrelu" : LeakyReLU,
    "tanh" : tanh,
    "sigmoid" : sigmoid,
    "softmax" : softmax,
}
