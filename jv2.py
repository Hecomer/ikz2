import numpy as np


def function(X, Y, Z):
    return 3*(X**2) + 4*(Y**2) + 5*(Z**2) + 2*X*Y - X*Z-2*Y*Z + X - 3*Z


def hooke_jeeves(X, Y, Z, step_size, tolerance, n):
    current_X = X
    current_Y = Y
    current_Z = Z
    best_X = current_X
    best_Y = current_Y
    best_Z = current_Z
    current_value = function(current_X, current_Y, current_Z)
    best_value = current_value

    while step_size > tolerance:
        flag = False
        n += 1
        for i, (dx, dy, dz) in enumerate([(step_size, 0, 0), (-step_size, 0, 0), (0, step_size, 0),
                                          (0, -step_size, 0), (0, 0, step_size),(0, 0, -step_size)]):
            new_X_pos = current_X + dx
            new_Y_pos = current_Y + dy
            new_Z_pos = current_Z + dz
            new_value_pos = function(new_X_pos, new_Y_pos, new_Z_pos)

            if new_value_pos < best_value:
                best_value = new_value_pos
                best_X = new_X_pos
                best_Y = new_Y_pos
                best_Z = new_Z_pos
                flag = True

        if not flag:
            step_size /= 2
        else:
            current_X = best_X
            current_Y = best_Y
            current_Z = best_Z

    return best_X, best_Y, best_Z, best_value, n


# Пример использования
initial_X = 1
initial_Y = -1
initial_Z = 1
step_size = 1
tolerance = 1e-6
n = 0
min_X, min_Y, min_Z, min_value, n = hooke_jeeves(initial_X, initial_Y, initial_Z, step_size, tolerance,n)
print("Минимальное значение функции:", min_value)
print("Точка минимума (X, Y, Z):", min_X, min_Y, min_Z)
print("Iterations:", n)
