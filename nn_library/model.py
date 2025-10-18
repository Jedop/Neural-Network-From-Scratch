try:
    import cupy as np
    print("GPU (CuPy) detected. Running on GPU.")
except ImportError:
    import numpy as np
    print("GPU (CuPy) not found. Running on CPU with NumPy.")
import numpy as npx 
from .activations import activation_functions, one_hot
import os

class NeuralNetwork():
    def __init__(self, alpha: float, activation: list = [], size : list =[10], input_size: int =28*28, layers: int =1, output_size: int =10):
        
        # Checking layers and size compatibility
        
        if layers > len(size):
            for i in range(layers - len(size)):
                size.append(100)
        elif layers < len(size):
            size = size [0: layers]
        
        size.insert(0, input_size)
        size.append(output_size)

        # Initialise all the variables

        self.t = 0
        self.beta_1 = 0.01
        self.beta_2 = 0.01
        self.size = size
        self.W = []
        self.b = []
        self.m = []
        self.v = []
        self.m_hat = []
        self.v_hat = []
        for i in range(layers+1):
            Wi = np.random.randn(size[i+1], size[i]) * np.sqrt(2 / size[i])
            mi = np.zeros((size[i+1], size[i]))
            vi = np.zeros((size[i+1], size[i]))
            bi = np.zeros((size[i+1], 1))  
            self.W.append(Wi)
            self.m.append(mi)
            self.v.append(vi)
            self.m_hat.append(mi)
            self.v_hat.append(vi)
            self.b.append(bi)
        self.alpha = alpha
        
        # Ensure activation is consistent with number of layers

        if activation is None or len(activation) == 0:
            self.activation = ["relu"] * (layers) + ["softmax"]
        else:
            self.activation = activation
            if len(self.activation) < layers + 1:
                self.activation += ["relu"] * (layers - len(self.activation) - 1)
                self.activation.append("softmax")
            elif len(self.activation) > layers + 1:
                self.activation = self.activation[0: layers + 1]
            self.activation[-1] = "softmax"  
            
    def forward_propagation(self, X):
        # Simply define a list for Z's and calculate all of them

        self.Z = []
        self.A = []
        for i in range(len(self.W)):
            Zi = np.dot(self.W[i], X) + self.b[i] if i == 0 else np.dot(self.W[i], self.A[i-1]) + self.b[i]
            self.Z.append(Zi)
            Ai = activation_functions[self.activation[i]](self.Z[i])
            self.A.append(Ai)
        return self.A[-1]

    
    def backward_propagation(self, X, Y):
        # Math implemented into code again
        self.t += 1
        m = X.shape[1]
        one_hot_Y = one_hot(Y)
        self.dZ = []
        self.dW = []
        self.db = []
        for i in range(len(self.A)):
            dZi = self.A[-(i+1)] - one_hot_Y if i == 0 else np.dot(self.W[-i].T, self.dZ[i-1]) * activation_functions[self.activation[-(i+1)]](self.Z[-(i+1)], 1)
            self.dZ.append(dZi)
            dWi = 1/m * np.dot(self.dZ[i], self.A[-(i+2)].T) if i + 2 <= len(self.A) else 1/m * np.dot(self.dZ[i], X.T)
            self.m[-(i+1)] = self.beta_1 * self.m[-(i+1)] + (1 - self.beta_1) * dWi
            self.v[-(i+1)] = self.beta_2 * self.v[-(i+1)] + (1 - self.beta_2) * dWi ** 2
            self.m_hat[-(i+1)] = (self.m[-(i+1)]) / (1 - self.beta_1 ** self.t)
            self.v_hat[-(i+1)] = (self.v[-(i+1)]) / (1 - self.beta_2 ** self.t)
            self.dW.append(dWi)
            dbi = 1/m * np.sum(self.dZ[i], 1, keepdims=True)
            self.db.append(dbi)
        self.dZ.reverse()
        self.dW.reverse()
        self.db.reverse()
    
    def update_parameters(self):
        # Adam Optimization
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - self.alpha * self.m_hat[i] / (np.sqrt(self.v_hat[i]) + 1e-8)
            self.b[i] = self.b[i] - self.alpha * self.db[i]

    def fit(self, epochs: int, batch_size: int, X, Y):
        for epoch in range(epochs):
            for i in range(0, X.shape[1], batch_size):
                X_batch = X[:, i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, Y_batch)
                self.update_parameters()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: train acc = {self.accuracy(X, Y):.4f}")

    def save(self, path="models/model_weights.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        weights_cpu = [w.get() for w in self.W]
        biases_cpu = [b.get() for b in self.b]

        npx.savez(
            path,
            weights=npx.array(weights_cpu, dtype=object),
            biases=npx.array(biases_cpu, dtype=object),
            size=self.size,
            alpha=self.alpha,
        )
        print(f"Model saved to {path}")

    def load(self, path="models/model_weights.npz"):
        data = npx.load(path, allow_pickle=True)
        self.W = [np.array(w) for w in data["weights"]]
        self.b = [np.array(b) for b in data["biases"]]
        self.size = list(data["size"])
        self.alpha = float(data["alpha"])
        print(f"Model loaded from {path}")

    def predict(self, X):
        A2 = self.forward_propagation(X)
        return np.argmax(A2, axis=0)
    
    def accuracy(self, X, Y):
        preds = self.predict(X)
        return np.mean(preds == Y)
