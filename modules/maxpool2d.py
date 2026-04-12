from modules.layer import Layer
#from cython_modules.maxpool2d import maxpool_forward_cython
import numpy as np
import os
import sys

_CY_MAXPOOL = None
try:
    _CY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cython_modules'))
    if _CY_DIR not in sys.path:
        sys.path.append(_CY_DIR)
    import optimizations as _cy_optimizations
    _CY_MAXPOOL = _cy_optimizations.maxpool2d_cython
except Exception:
    _CY_MAXPOOL = None

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        if _CY_MAXPOOL is not None:
            self.input = np.asarray(input)
            cy_in = np.ascontiguousarray(self.input, dtype=np.float32)
            output = _CY_MAXPOOL(cy_in, int(self.kernel_size), int(self.stride))

            # Conservamos indices para backward con la misma logica actual.
            B, C, H, W = self.input.shape
            KH, KW = self.kernel_size, self.kernel_size
            SH, SW = self.stride, self.stride
            out_h = (H - KH) // SH + 1
            out_w = (W - KW) // SW + 1

            windows = np.lib.stride_tricks.sliding_window_view(self.input, (KH, KW), axis=(2, 3))
            windows = windows[:, :, ::SH, ::SW, :, :]
            flat_windows = windows.reshape(B, C, out_h, out_w, KH * KW)
            max_pos = np.argmax(flat_windows, axis=-1)

            local_r = max_pos // KW
            local_s = max_pos % KW
            base_r = (np.arange(out_h, dtype=np.int64) * SH).reshape(1, 1, out_h, 1)
            base_s = (np.arange(out_w, dtype=np.int64) * SW).reshape(1, 1, 1, out_w)
            global_r = base_r + local_r
            global_s = base_s + local_s
            self.max_indices = np.stack((global_r, global_s), axis=-1)
            return output.astype(self.input.dtype, copy=False)

        # --- INICIO BLOQUE GENERADO CON IA ---
        # Vectorizacion con vistas por stride: calcula maximos e indices sin bucles por ventana.
        self.input = np.asarray(input)
        B, C, H, W = input.shape
        KH, KW = self.kernel_size, self.kernel_size
        SH, SW = self.stride, self.stride

        out_h = (H - KH) // SH + 1
        out_w = (W - KW) // SW + 1

        windows = np.lib.stride_tricks.sliding_window_view(self.input, (KH, KW), axis=(2, 3))
        windows = windows[:, :, ::SH, ::SW, :, :]
        flat_windows = windows.reshape(B, C, out_h, out_w, KH * KW)

        max_pos = np.argmax(flat_windows, axis=-1)
        output = np.max(flat_windows, axis=-1)

        local_r = max_pos // KW
        local_s = max_pos % KW

        base_r = (np.arange(out_h, dtype=np.int64) * SH).reshape(1, 1, out_h, 1)
        base_s = (np.arange(out_w, dtype=np.int64) * SW).reshape(1, 1, 1, out_w)
        global_r = base_r + local_r
        global_s = base_s + local_s
        self.max_indices = np.stack((global_r, global_s), axis=-1)
        # --- FIN BLOQUE GENERADO CON IA ---

        # Codigo anterior: Los 3 bucles anidados son lentos.
        # self.input = input
        # B, C, H, W = input.shape
        # KH, KW = self.kernel_size, self.kernel_size
        # SH, SW = self.stride, self.stride
        # out_h = (H - KH) // SH + 1
        # out_w = (W - KW) // SW + 1
        # self.max_indices = np.zeros((B, C, out_h, out_w, 2), dtype=int)
        # output = np.zeros((B, C, out_h, out_w),dtype=input.dtype)
        # for b in range(B):
        #     for c in range(C):
        #         for i in range(out_h):
        #             for j in range(out_w):
        #                 h_start = i * SH
        #                 h_end = h_start + KH
        #                 w_start = j * SW
        #                 w_end = w_start + KW
        #                 window = input[b, c, h_start:h_end, w_start:w_end]
        #                 max_idx = np.unravel_index(np.argmax(window), window.shape)
        #                 max_val = window[max_idx]
        #                 output[b, c, i, j] = max_val
        #                 self.max_indices[b, c, i, j] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def backward(self, grad_output, learning_rate=None):
        # --- INICIO BLOQUE GENERADO CON IA ---
        # Retropropagacion vectorizada: reparte gradientes a los indices maximos con np.add.at.
        B, C, H, W = self.input.shape
        grad_output = np.asarray(grad_output)
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        b_idx = np.arange(B, dtype=np.int64).reshape(B, 1, 1, 1)
        c_idx = np.arange(C, dtype=np.int64).reshape(1, C, 1, 1)
        b_idx = np.broadcast_to(b_idx, (B, C, out_h, out_w))
        c_idx = np.broadcast_to(c_idx, (B, C, out_h, out_w))

        r_idx = self.max_indices[..., 0]
        s_idx = self.max_indices[..., 1]
        np.add.at(grad_input, (b_idx, c_idx, r_idx, s_idx), grad_output)
        # --- FIN BLOQUE GENERADO CON IA ---

        # Codigo anterior: Los 3 bucles anidados son lentos.
        # B, C, H, W = self.input.shape
        # grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        # out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        # for b in range(B):
        #     for c in range(C):
        #         for i in range(out_h):
        #             for j in range(out_w):
        #                 r, s = self.max_indices[b, c, i, j]
        #                 grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input