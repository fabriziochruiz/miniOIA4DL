from modules.layer import Layer
from modules.utils import *
#from cython_modules.im2col import im2col_forward_cython

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_algo = conv_algo
        
        # MODIFICAR: Añadir nuevo if-else para otros algoritmos de convolución
        if conv_algo == 0:
            self.mode = 'direct'
        elif conv_algo == 1:
            self.mode = 'im2col'
        elif conv_algo == 2:
            print("Algoritmo 2 (im2col fused) no implementado aun, usando modo direct")
            self.mode = 'direct'
        else:
            print(f"Algoritmo {conv_algo} no soportado, usando modo direct")
            self.mode = 'direct'

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
    
    def forward(self, input, training=True):
        self.input = input
        # PISTA: Usar estos if-else si implementas más algoritmos de convolución
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col':
            return self._forward_im2col(input)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    def backward(self, grad_output, learning_rate):
        # Para mantener compatibilidad con entrenamiento, reutilizamos backward directo.
        if self.mode in ('direct', 'im2col'):
            return self._backward_direct(grad_output, learning_rate)
        raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---

    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[b, in_c,
                                           i * self.stride:i * self.stride + k_h,
                                           j * self.stride:j * self.stride + k_w]
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                output[b, out_c] += self.biases[out_c]

        return output

    # --- IM2COL IMPLEMENTATIONS ---

    def _extract_patches_as_matrix(self, input):
        # --- INICIO BLOQUE GENERADO CON IA ---
        # im2col con vistas por stride y layout contiguo para mejorar acceso a memoria en GEMM.
        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)
        else:
            input = input.astype(np.float32, copy=False)

        k_h, k_w = self.kernel_size, self.kernel_size
        windows = sliding_window_view(input, (k_h, k_w), axis=(2, 3))
        windows = windows[:, :, ::self.stride, ::self.stride, :, :]

        batch_size = input.shape[0]
        out_h = windows.shape[2]
        out_w = windows.shape[3]

        patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_h * out_w, -1)
        patches = np.ascontiguousarray(patches, dtype=np.float32)

        # Codigo anterior: Los 3 bucles anidados son lentos.
        # if self.padding > 0:
        #     input = np.pad(input,
        #                    ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
        #                    mode='constant').astype(np.float32)
        # else:
        #     input = input.astype(np.float32, copy=False)
        # k_h, k_w = self.kernel_size, self.kernel_size
        # windows = sliding_window_view(input, (k_h, k_w), axis=(2, 3))
        # windows = windows[:, :, ::self.stride, ::self.stride, :, :]
        # batch_size = input.shape[0]
        # out_h = windows.shape[2]
        # out_w = windows.shape[3]
        # patches = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size * out_h * out_w, -1)
        # --- FIN BLOQUE GENERADO CON IA ---
        return patches, batch_size, out_h, out_w

    def _forward_im2col(self, input):
        # --- INICIO BLOQUE GENERADO CON IA ---
        # Convolucion como GEMM: im2col(X) @ W + b.
        patches, batch_size, out_h, out_w = self._extract_patches_as_matrix(input)
        kernels_2d = np.ascontiguousarray(self.kernels.reshape(self.out_channels, -1).T, dtype=np.float32)
        output_2d = patches @ kernels_2d
        output_2d += self.biases
        output = output_2d.reshape(batch_size, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

        # Codigo anterior: Los 3 bucles anidados son lentos.
        # patches, batch_size, out_h, out_w = self._extract_patches_as_matrix(input)
        # kernels_2d = self.kernels.reshape(self.out_channels, -1).T
        # output_2d = patches @ kernels_2d
        # output_2d += self.biases
        # output = output_2d.reshape(batch_size, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        # --- FIN BLOQUE GENERADO CON IA ---
        return output.astype(np.float32, copy=False)

    def _backward_direct(self, grad_output, learning_rate):
        # --- INICIO BLOQUE GENERADO CON IA ---
        # Retropropagacion con reordenacion de bucles para mejor localidad de cache en nivel basico.
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input.astype(np.float32, copy=False)

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.sum(grad_output, axis=(0, 2, 3), keepdims=False)

        for b in range(batch_size):
            for i in range(out_h):
                for j in range(out_w):
                    r = i * self.stride
                    c = j * self.stride
                    for out_c in range(self.out_channels):
                        for in_c in range(self.in_channels):
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]

        # Codigo anterior: Los 5 bucles anidados con peor localidad de cache.
        # for b in range(batch_size):
        #     for out_c in range(self.out_channels):
        #         for in_c in range(self.in_channels):
        #             for i in range(out_h):
        #                 for j in range(out_w):
        #                     r = i * self.stride
        #                     c = j * self.stride
        #                     region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
        #                     grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
        #                     grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
        #             grad_biases[out_c] += np.sum(grad_output[b, out_c])
        # --- FIN BLOQUE GENERADO CON IA ---

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

    # PISTA: Se te ocurren otros algoritmos de convolución?



    