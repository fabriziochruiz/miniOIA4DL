# INICIO BLOQUE GENERADO CON IA
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp

# Declaramos los tipos de datos para máxima velocidad
ctypedef cnp.float32_t DTYPE_t

def im2col_cython(DTYPE_t[:, :, :, :] x, int filter_h, int filter_w, int stride, int pad):
    """
    Transforma un batch de imágenes (N, C, H, W) a una matriz de columnas (N, C*F*F, H_out*W_out)
    para acelerar la convolución mediante multiplicación de matrices.
    """
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int H_out = (H + 2 * pad - filter_h) // stride + 1
    cdef int W_out = (W + 2 * pad - filter_w) // stride + 1
    
    # Creamos el array de salida en NumPy y obtenemos su vista en memoria C
    cdef cnp.ndarray[DTYPE_t, ndim=3] cols = np.zeros((N, C * filter_h * filter_w, H_out * W_out), dtype=np.float32)
    cdef DTYPE_t[:, :, :] cols_view = cols
    
    cdef int n, c, i, j, row, col, ii, jj, p_row, p_col
    cdef int col_idx
    
    for n in range(N):
        for c in range(C):
            for i in range(filter_h):
                for j in range(filter_w):
                    row = c * filter_h * filter_w + i * filter_w + j
                    for ii in range(H_out):
                        for jj in range(W_out):
                            p_row = ii * stride - pad + i
                            p_col = jj * stride - pad + j
                            
                            col_idx = ii * W_out + jj
                            
                            # Comprobamos los límites del padding
                            if p_row >= 0 and p_row < H and p_col >= 0 and p_col < W:
                                cols_view[n, row, col_idx] = x[n, c, p_row, p_col]
                                
    return cols

def maxpool2d_cython(DTYPE_t[:, :, :, :] x, int pool_size, int stride):
    """
    Operación MaxPool2D altamente optimizada usando bucles puros en C.
    """
    cdef int N = x.shape[0]
    cdef int C = x.shape[1]
    cdef int H = x.shape[2]
    cdef int W = x.shape[3]
    
    cdef int H_out = (H - pool_size) // stride + 1
    cdef int W_out = (W - pool_size) // stride + 1
    
    cdef cnp.ndarray[DTYPE_t, ndim=4] out = np.zeros((N, C, H_out, W_out), dtype=np.float32)
    cdef DTYPE_t[:, :, :, :] out_view = out
    
    cdef int n, c, out_h, out_w, ph, pw
    cdef DTYPE_t max_val, current_val
    
    for n in range(N):
        for c in range(C):
            for out_h in range(H_out):
                for out_w in range(W_out):
                    max_val = -float('inf')
                    for ph in range(pool_size):
                        for pw in range(pool_size):
                            current_val = x[n, c, out_h * stride + ph, out_w * stride + pw]
                            if current_val > max_val:
                                max_val = current_val
                    out_view[n, c, out_h, out_w] = max_val
                    
    return out
# FIN BLOQUE GENERADO CON IA