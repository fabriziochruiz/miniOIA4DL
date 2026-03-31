from modules.layer import Layer
# from modules.utils import *  # Version anterior: matmul manual con bucles

import numpy as np

class Dense(Layer):
    def __init__(self, in_features, out_features,weight_init="he"):
        self.in_features = in_features
        self.out_features = out_features

        if weight_init == "he":
            std = np.sqrt(2.0 / in_features)
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * std
        elif weight_init == "custom":
            self.weights = np.zeros((in_features, out_features), dtype=np.float32)
        else:
            self.weights = np.random.randn(in_features, out_features).astype(np.float32) * (1 / in_features**0.5)

        self.biases = np.zeros(out_features, dtype=np.float32)

        self.input = None

    def forward(self, input, training=True):  # input: [batch_size x in_features]
        # Optimizacion basica: usar operaciones vectorizadas de NumPy
        
        # --- INICIO BLOQUE GENERADO CON IA ---
        self.input = np.asarray(input, dtype=np.float32)
        output = self.input @ self.weights + self.biases
        # --- FIN BLOQUE GENERADO CON IA ---
        
        # Version anterior (mas lenta):
        # batch_size = self.input.shape[0]
        # output = np.zeros((batch_size, self.out_features), dtype=np.float32)
        # output = matmul_biasses(self.input, self.weights, output, self.biases)
        self.output = output
        return output

    def backward(self, grad_output, learning_rate):
        grad_output = np.asarray(grad_output, dtype=np.float32)
        
        # Vectorizacion de gradiantes para un mejor rendimiento en el nivel básico de NumPy.
        # Ya que los 3 bucles anidados son muy costosos, por eso se usa la vectorizacion.

        # --- INICIO BLOQUE GENERADO CON IA ---
        grad_weights = self.input.T @ grad_output
        # --- FIN BLOQUE GENERADO CON IA ---

        # Version anterior (mas lenta): gradientes con 3 bucles anidados.
        #for i in range(self.in_features):
        #    for j in range(self.out_features):
        #        for b in range(batch_size):
        #            grad_weights[i][j] += self.input[b][i] * grad_output[b][j]
        
        # --- INICIO BLOQUE GENERADO CON IA ---
        grad_biases = np.sum(grad_output, axis=0)
        grad_input = grad_output @ self.weights.T
        # --- FIN BLOQUE GENERADO CON IA ---

        # Version anterior (mas lenta): gradientes con 3 bucles anidados.
        #for b in range(batch_size):
        #    for i in range(self.in_features):
        #        for j in range(self.out_features):
        #            grad_input[b][i] += grad_output[b][j] * self.weights[i][j
        
        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_input
    
    def get_weights(self):
        return {'weights': self.weights, 'biases': self.biases}

    def set_weights(self, weights):
        self.weights = weights['weights']
        self.biases = weights['biases']