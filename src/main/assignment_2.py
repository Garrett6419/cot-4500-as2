import numpy as np  # type: ignore

# Neville's Method for polynomial interpolation
def neville_method(x_values, y_values, x):
    n = len(x_values)
    Q = np.zeros((n, n))
    
    for i in range(n):
        Q[i][0] = y_values[i]
    
    for j in range(1, n):
        for i in range(n - j):
            Q[i][j] = ((x - x_values[i + j]) * Q[i][j - 1] + (x_values[i] - x) * Q[i + 1][j - 1]) / (x_values[i] - x_values[i + j]) # goodness this was a really long line and oh no I'm just making it longer and longer and
    
    return Q[0][n - 1]

# Newton's Forward Method for polynomial approximation
def newtons_forward(x_values, y_values):
    n = len(x_values)
    table = np.zeros((n, n))
    table[:, 0] = y_values
    
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = table[i + 1][j - 1] - table[i][j - 1]
    
    return table

# Approximate f(7.3) using the results from Newton's Forward Method
def approximate_f_using_newton(table, x_values, x):
    n = len(x_values)
    h = x_values[1] - x_values[0]
    p = (x - x_values[0]) / h
    result = table[0][0]
    
    for i in range(1, n):
        term = table[0][i]
        for j in range(i):
            term *= (p - j) / (i - j)
        result += term
    
    return result

# Divided Difference Method for Hermite polynomial approximation
def divided_difference(x_values, y_values, derivatives):
    n = len(x_values)
    table = np.zeros((n, n))
    table[:, 0] = y_values
    
    for j in range(1, n):
        for i in range(n - j):
            if j == 1:
                table[i][j] = derivatives[i]
            else:
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x_values[i + j] - x_values[i])
    
    return table

# Cubic Spline Interpolation
def cubic_spline(x_values, y_values):
    n = len(x_values)
    if n < 2:
        raise ValueError("At least two points are required for cubic spline interpolation.")
    
    # Ensure x_values are unique and sorted
    if len(set(x_values)) != n:
        raise ValueError("x_values must be unique.")
    
    # Step sizes
    h = np.diff(x_values)
    
    # Initialize matrix A and vector b
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Set up the system of equations
    # Natural spline conditions
    A[0][0] = 1  # S''(x_0) = 0
    A[-1][-1] = 1  # S''(x_n) = 0

    for i in range(1, n - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
        b[i] = 3 * ((y_values[i + 1] - y_values[i]) / h[i] - (y_values[i] - y_values[i - 1]) / h[i - 1])

    # Solve for the second derivatives (c coefficients)
    c = np.linalg.solve(A, b)

    return A, b, c  # Return A, b, c

# Example usage of the algorithms
if __name__ == "__main__":
    # Task 1: Neville's Method
    x_values_1 = [3.6, 3.8, 3.9]
    y_values_1 = [1.675, 1.436, 1.318]
    print("Neville's Method Result for f(3.7):", neville_method(x_values_1, y_values_1, 3.7))
    print("")

    # Task 2: Newton's Forward Method
    x_values_2 = [7.2, 7.4, 7.5 , 7.6]
    y_values_2 = [23.5492, 25.3913, 26.8224, 27.4589]
    table = newtons_forward(x_values_2, y_values_2)
    print("Newton's Forward Method Table:\n", table)
    print("")

    # Task 3: Approximate f(7.3) using Newton's Forward Method
    f_approx = approximate_f_using_newton(table, x_values_2, 7.3)
    print("Approximation of f(7.3):", f_approx)
    print("")

    # Task 4: Divided Difference Method
    x_values_3 = [3.6, 3.8, 3.9]
    y_values_3 = [1.675, 1.436, 1.318]
    derivatives = [-1.195, -1.188, -1.182]
    hermite_table = divided_difference(x_values_3, y_values_3, derivatives)
    print("Hermite Polynomial Approximation Matrix:\n", hermite_table)
    print("")

    # Task 5: Cubic Spline Interpolation
    x_values_4 = [2, 5, 8, 10]  
    y_values_4 = [3, 5, 7, 9]  
    A, b, x = cubic_spline(x_values_4, y_values_4)
    print("Matrix A:\n", A)
    print("Vector b:\n", b)
    print("Vector x:\n", x)