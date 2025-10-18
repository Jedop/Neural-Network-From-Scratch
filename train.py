try:
    import cupy as np
    print("GPU (CuPy) detected. Running on GPU.")
except ImportError:
    import numpy as np
    print("GPU (CuPy) not found. Running on CPU with NumPy.")
from nn_library.model import NeuralNetwork
import argparse
from utils import load_data, shape_data 

def train_data(x_train1, x_test1, y_train, y_test, epochs=41, batch_size=256, lr=0.001, output_size=10, layers=2, size=[100, 100], activation=[]):
    if layers > len(size):
        for i in range(layers - len(size)):
            size.append(100)
    elif layers < len(size):
        size = size [0: layers]
    print(layers, size)
    nn = NeuralNetwork(input_size=x_train1.shape[0], size=size, alpha=lr, output_size=output_size, layers=layers, activation=activation)

    for epoch in range(epochs):
        for i in range(0, x_train1.shape[1], batch_size):
            X_batch = x_train1[:, i:i+batch_size]
            Y_batch = y_train[i:i+batch_size]
            nn.forward_propagation(X_batch)
            nn.backward_propagation(X_batch, Y_batch)
            nn.update_parameters()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train acc = {nn.accuracy(x_train1, y_train):.4f}")

    nn.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on the MNIST dataset.')
    parser.add_argument('--epochs', type=int, default=41, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--output_size', type=int, default=10, help='Output Size')
    parser.add_argument('--layers', type=int, default=2, help='Number of Hidden Layers')
    parser.add_argument('--size', nargs='+', type=int, default=[100, 100], help='Size of each hidden layer in the form of a list. If number of layers is lower than size list, it will be truncated. If higher, the unspecified layers will get a default value of 100')
    parser.add_argument('--activation', nargs='+', type=str, default=["relu", "relu", "softmax"], help="Per layer activation function, including output layer. Output layer will get softmax by default. Unspecified layers will get ReLU by default.")
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_data()
    x_train1, y_train, x_test1, y_test = shape_data(x_train, y_train, x_test, y_test)

    # Train
    train_data(
        x_train1=x_train1,
        x_test1=x_test1,
        y_train=y_train,
        y_test=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_size=args.output_size,
        layers=args.layers,
        size=args.size,
        activation=args.activation,
    )
