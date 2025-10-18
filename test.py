try:
    import cupy as np
    print("GPU (CuPy) detected. Running on GPU.")
except ImportError:
    import numpy as np
    print("GPU (CuPy) not found. Running on CPU with NumPy.")
import struct
from array import array
from os.path  import join
from nn_library.model import NeuralNetwork
import argparse
from utils import load_data, shape_data  

def test_data(x_test1, y_test, path=None):
    nn = NeuralNetwork(size=[100, 100], layers=2, alpha=0.001)
    if path == None:
        nn.load()
    else:
        nn.load(path)
    print("Test accuracy:", nn.accuracy(x_test1, y_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test your neural network on the MNIST dataset.')
    parser.add_argument('--path_to_model', type=str, default="models/model_weights.npz", help='Path to your model')
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_data()
    x_train1, y_train, x_test1, y_test = shape_data(x_train, y_train, x_test, y_test)

    test_data(x_test1, y_test, path=args.path_to_model)
