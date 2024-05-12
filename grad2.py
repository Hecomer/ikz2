import numpy as np


def function_to_minimize(X, Y, Z):
    return 3*(X**2) + 4*(Y**2) + 5*(Z**2) + 2*X*Y - X*Z-2*Y*Z + X - 3*Z


def gradient_function(X, Y, Z):
    gradient_X = 6*X + 2*Y - Z + 1
    gradient_Y = 8*Y + 2*X - 2*Z
    gradient_Z = 10*Z - X - 2*Y - 3
    return gradient_X, gradient_Y, gradient_Z


def gradient_descent(X, Y, Z, alpha, tolerance, n):
    current_X = X
    current_Y = Y
    current_Z = Z
    current_value = function_to_minimize(current_X, current_Y, current_Z)

    while True:
        gradient_X, gradient_Y, gradient_Z = gradient_function(current_X, current_Y, current_Z)

        new_X = current_X - alpha * gradient_X
        new_Y = current_Y - alpha * gradient_Y
        new_Z = current_Z - alpha * gradient_Z

        new_value = function_to_minimize(new_X, new_Y, new_Z)

        if abs(new_value - current_value) < tolerance:
            break

        n += 1
        current_X = new_X
        current_Y = new_Y
        current_Z = new_Z
        current_value = new_value

    return current_X, current_Y, current_Z, current_value, n


# Пример использования
initial_X = 1
initial_Y = -1
initial_Z = 1
alpha = 0.01  # Adjusted learning rate
tolerance = 1e-5
n = 0

min_X, min_Y, min_Z, min_value, n = gradient_descent(initial_X, initial_Y, initial_Z, alpha, tolerance, n)

print("Минимальное значение функции:", min_value)
print("Точка минимума (X, Y, Z):", min_X, min_Y, min_Z)
print("Iterations:", n)

