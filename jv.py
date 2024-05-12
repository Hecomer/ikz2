import numpy as np


def function(X, Y):
    return 7*(X**2) + 2*X*Y + 5*(Y**2) + X - 10*Y


def hooke_jeeves(initial_x, initial_y, step_size, tolerance,n):
    current_x = initial_x
    current_y = initial_y
    best_x = current_x
    best_y = current_y
    current_value = function(current_x, current_y)
    best_value = current_value

    while step_size > tolerance:
        flag = False
        n += 1
        for i, (dx, dy) in enumerate([(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]):
            new_x_pos = current_x + dx
            new_y_pos = current_y + dy
            new_value_pos = function(new_x_pos, new_y_pos)

            if new_value_pos < best_value:
                best_value = new_value_pos
                best_x = new_x_pos
                best_y = new_y_pos
                flag = True

        if not flag:
            step_size /= 2
        else:
            current_x = best_x
            current_y = best_y

    return best_x, best_y, best_value, n


initial_x = 1
initial_y = -1
step_size = 1
tolerance = 1e-6
n = 0
min_x, min_y, min_value, n = hooke_jeeves(initial_x, initial_y, step_size, tolerance,n)
print("Минимальное значение функции:", min_value)
print("Точка минимума (x, y):", min_x, min_y)
print("Iterations:", n)
