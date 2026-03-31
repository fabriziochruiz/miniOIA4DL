#PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses(A, B, C, bias):
    # --- INICIO BLOQUE GENERADO CON IA ---
    # Version vectorizada: usa BLAS de NumPy para calcular A @ B y sumar bias por fila.
    C[:] = A @ B + bias
    # --- FIN BLOQUE GENERADO CON IA ---

    # Codigo anterior: Los 3 bucles anidados son lentos.
    # m, p, n = A.shape[0], A.shape[1], B.shape[1]
    # for i in range(m):
    #     for j in range(n):
    #         for k in range(p):
    #             C[i][j] += A[i][k] * B[k][j]
    #         C[i][j] += bias[j]
    return C

