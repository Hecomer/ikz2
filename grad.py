import numpy as np


# Определение функции
def function_to_minimize(X, Y):
    return 7*(X**2) + 2*X*Y + 5*(Y**2) + X - 10*Y


# Определение градиента функции
def gradient(X, Y):
    df_dx = 14*X + 2*Y + 1
    df_dy = 2*X + 10*Y - 10
    return np.array([df_dx, df_dy])


# Метод градиентного спуска
def gradient_descent(initial_point, alpha, tolerance, n):
    current_point = np.array(initial_point, dtype=float)
    current_value = function_to_minimize(*current_point)

    while True:
        grad = gradient(*current_point)
        new_point = current_point - alpha * grad
        new_value = function_to_minimize(*new_point)
        if abs(new_value - current_value) < tolerance:
            break
        current_point = new_point
        current_value = new_value
        n += 1
    return current_point, current_value, n


initial_point = [1, 1]
alpha = 0.1
tolerance = 1e-5
n = 0
min_point, min_value, n = gradient_descent(initial_point, alpha, tolerance, n)

print("Минимальное значение функции:", min_value)
print("Точка минимума:", min_point)
print("Количество итераций:", n)