import argparse
try:
    import cupy as np
    print("GPU (CuPy) detected. Running on GPU.")
except ImportError:
    import numpy as np
    print("GPU (CuPy) not found. Running on CPU with NumPy.")
import numpy as npx 
import matplotlib.pyplot as plt
from nn_library.model import NeuralNetwork
from utils import load_data, shape_data 

def visualize_predictions(x_test, y_test, path, num_samples=5):
    nn = NeuralNetwork(size=[100, 100], layers=2, alpha=0.001)
    nn.load(path)

    # Forward pass
    y_pred = nn.forward_propagation(x_test)
    preds = np.argmax(y_pred, axis=0)

    # Determine how y_test is shaped
    if y_test.ndim == 2:  # one-hot encoded (10, N) or (N, 10)
        if y_test.shape[0] == 10:
            labels = np.argmax(y_test, axis=0)
        else:
            labels = np.argmax(y_test, axis=1)
    else:  # already 1D integer labels
        labels = y_test

    # Pick random samples
    indices = np.random.choice(x_test.shape[1], num_samples, replace=False)

    preds_cpu = np.asnumpy(preds)
    labels_cpu = np.asnumpy(labels)
    preds_cpu = npx.ravel(preds_cpu)
    labels_cpu = npx.ravel(labels_cpu)

    for i, idx in enumerate(indices.get()):
        img = np.asnumpy(x_test[:, idx]).reshape(28, 28)

        pred_val = int(preds_cpu[idx])
        true_val = int(labels_cpu[idx])

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"P: {pred_val}\nT: {true_val}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize predictions of a trained model.')
    parser.add_argument('--path_to_model', type=str, default='models/model_weights.npz',
                        help='Path to the trained model weights.')
    args = parser.parse_args()

    # Load and prepare data
    x_train, y_train, x_test, y_test = load_data()
    x_train1, y_train, x_test1, y_test = shape_data(x_train, y_train, x_test, y_test)

    # Visualize predictions
    visualize_predictions(x_test1, y_test, args.path_to_model)
