import numpy as np


def J(X, Y, Z):
    return 3*(X**2) + 4*(Y**2) + 5*(Z**2) + 2*X*Y - X*Z-2*Y*Z + X - 3*Z


def gradient_J(X, Y, Z):
    # Частные производные функции J(x, y, z) по переменным x, y и z
    gradient_X = 6 * X + 2 * Y - Z + 1
    gradient_Y = 8 * Y + 2 * X - 2 * Z
    gradient_Z = 10 * Z - X - 2 * Y - 3
    return gradient_X, gradient_Y, gradient_Z


def bisection_method(func, a, b, tol=1e-5):
    while (b - a) / 2 > tol:
        x1 = (a + b - tol) / 2
        x2 = (a + b + tol) / 2
        if func(x1) < func(x2):
            b = x2
        else:
            a = x1
    return (a + b) / 2


def steepest_descent(x_init, y_init, z_init, alpha_init, eps=1e-5):
    x_current = x_init
    y_current = y_init
    z_current = z_init
    n = 0

    while np.sqrt(gradient_J(x_current, y_current, z_current)[0] ** 2 +
                   gradient_J(x_current, y_current, z_current)[1] ** 2 +
                   gradient_J(x_current, y_current, z_current)[2] ** 2) >= eps:
        # Вычисляем градиент функции J(x, y, z) в текущей точке
        grad_x, grad_y, grad_z = gradient_J(x_current, y_current, z_current)

        # Начальное значение альфа на текущей итерации
        alpha = alpha_init

        # Поиск оптимального шага методом дихотомии
        alpha_opt = bisection_method(lambda alpha: J(x_current - alpha * grad_x, y_current - alpha * grad_y, z_current - alpha * grad_z), 0, alpha)
        # Обновляем значения переменных x, y и z
        x_current -= alpha_opt * grad_x
        y_current -= alpha_opt * grad_y
        z_current -= alpha_opt * grad_z

        # Увеличиваем счетчик итераций
        n += 1

    return x_current, y_current, z_current, n


# Начальные значения переменных x, y, z и начальное значение альфа
x_init = 1.0
y_init = -1.0
z_init = 1.0
alpha_init = 0.1

# Запускаем метод наискорейшего спуска
x_min, y_min, z_min, iterations = steepest_descent(x_init, y_init, z_init, alpha_init)
print("Минимальное значение (x, y, z) = ({:.4f}, {:.4f}, {:.4f}), J(x, y, z) = {:.5f}".format(x_min, y_min, z_min, J(x_min, y_min, z_min)))
print("Количество итераций:", iterations)
