"""
# Three-Layer Feedforward Neural Network from Scratch

This implementation builds a fully connected feedforward neural network using only NumPy, without any deep learning frameworks. The network architecture follows a 784 → 128 → 64 → 10 design, where the input layer takes flattened 28x28 MNIST images, two hidden layers with 128 and 64 neurons respectively use ReLU activation to introduce non-linearity, and the output layer with 10 neurons uses Softmax to produce class probabilities for digits 0-9.

All gradients are derived from first principles. The weight gradients for each layer are:

- dL/dW3 = A2 · (A3 - Y)
- dL/dW2 = (A3 - Y) · W3 · ReLU'(Z2) · A1
- dL/dW1 = (A3 - Y) · W3 · ReLU'(Z2) · W2 · ReLU'(Z1) · X

Where ReLU'(Z) = 1 if Z > 0, else 0. Weight matrices are stored as separate 2D NumPy arrays — W1 (784, 128), W2 (128, 64), W3 (64, 10) — since differing dimensions prevent stacking into a single 3D array.

## Variable Shapes

### Weights & Biases
| Variable | Shape | Reason |
|----------|-------|--------|
| W1 | (784, 128) | 784 inputs → 128 neurons |
| W2 | (128, 64)  | 128 neurons → 64 neurons |
| W3 | (64, 10)   | 64 neurons → 10 output classes |
| b1 | (128,)     | one bias per neuron in layer 1 |
| b2 | (64,)      | one bias per neuron in layer 2 |
| b3 | (10,)      | one bias per output class |

### Forward Pass
| Variable           | Shape | How |
|--------------------|---------|-----|
| X                  | (n, 784)| n samples, 784 pixels each |
| Y                  | (n, 10) | one-hot encoded labels |
| Z1 = X @ W1 + b1   | (n, 784) @ (784, 128) → (n, 128) | |
| A1 = ReLU(Z1)      | (n, 128) | same shape as Z1 |
| Z2 = A1 @ W2 + b2  | (n, 128) @ (128, 64) → (n, 64)  | |
| A2 = ReLU(Z2)      | (n, 64)  | same shape as Z2 |
| A3 = Softmax(A2 @ W3 + b3) | (n, 64) @ (64, 10) → (n, 10) | |

### Backward Pass
| Variable | Shape | How |
|----------|-------|-----|
| dZ3 = A3 - Y                 | (n, 10)                             | (n, 10) - (n, 10) |
| Grad_W3 = A2.T @ dZ3 / n     | (64, n) @ (n, 10) → **(64, 10)**    | matches W3        |
| Grad_b3 = sum(dZ3) / n       | (10,)                               | sum across n samples |
| dZ2 = dZ3 @ W3.T * ReLU'(Z2) | (n, 10) @ (10, 64) → **(n, 64)**    | masked by (Z2 > 0) |
| Grad_W2 = A1.T @ dZ2 / n     | (128, n) @ (n, 64) → **(128, 64)**  | matches W2 |
| Grad_b2 = sum(dZ2) / n       | (64,)                               | sum across n samples    |
| dZ1 = dZ2 @ W2.T * ReLU'(Z1) | (n, 64) @ (64, 128) → **(n, 128)**  | masked by (Z1 > 0) |
| Grad_W1 = X.T @ dZ1 / n      | (784, n) @ (n, 128) → **(784, 128)**| matches W1 |
| Grad_b1 = sum(dZ1) / n       | (128,)                              | sum across n samples |
"""

import numpy as np
import pandas as pd

class Neural_Network:
    def __init__(self):
        self.W1 = np.random.randn(784, 128) * 0.01
        self.W2 = np.random.randn(128, 64) * 0.01
        self.W3 = np.random.randn(64, 10) * 0.01
        self.b1 = np.full((128,), 0.1)
        self.b2 = np.full((64,), 0.1)
        self.b3 = np.full((10,), 0.1)

    def softmax(self, arr):
        arr -= np.max(arr, axis=1, keepdims=True)
        logits = np.exp(arr)
        return logits / np.sum(logits, axis=1, keepdims=True)

    def fit(self, X, Y, iters=1000, alpha=0.0001,lamb=0.01):
        n = X.shape[0]
        for i in range(iters):
            Z1 = X @ self.W1 + self.b1
            A1 = np.maximum(0, Z1)
            Z2 = A1 @ self.W2 + self.b2
            A2 = np.maximum(0, Z2)
            A3 = self.softmax(A2 @ self.W3 + self.b3)

            if i % 100 == 0:
                loss = -np.sum(Y * np.log(np.clip(A3, 1e-15, 1))) / n
                print(f"iter {i} | loss: {loss:.4f}")

            dZ3 = A3 - Y
            Grad_W3 = (A2.T @ dZ3 / n) + (lamb*self.W3/n)
            Grad_b3 = np.sum(dZ3, axis=0) / n

            dZ2 = dZ3 @ self.W3.T * (Z2 > 0)
            Grad_W2 = A1.T @ dZ2 / n + (lamb*self.W2/n)
            Grad_b2 = np.sum(dZ2, axis=0) / n

            dZ1 = dZ2 @ self.W2.T * (Z1 > 0)
            Grad_W1 = X.T @ dZ1 / n + (lamb*self.W1/n)
            Grad_b1 = np.sum(dZ1, axis=0) / n

            self.W3 -= alpha * Grad_W3
            self.b3 -= alpha * Grad_b3
            self.W2 -= alpha * Grad_W2
            self.b2 -= alpha * Grad_b2
            self.W1 -= alpha * Grad_W1
            self.b1 -= alpha * Grad_b1

    def predict(self, X):
        A1 = np.maximum(0, X @ self.W1 + self.b1)
        A2 = np.maximum(0, A1 @ self.W2 + self.b2)
        A3 = self.softmax(A2 @ self.W3 + self.b3)
        return np.argmax(A3, axis=1)
    