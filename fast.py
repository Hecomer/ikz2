import numpy as np


def J(X, Y):
    return 7*(X**2) + 2*X*Y + 5*(Y**2) + X - 10*Y


def gradient_J(X, Y):
    # Частные производные функции J(x, y) по переменным x и y
    partial_x = 14*X + 2*Y + 1
    partial_y = 2*X + 10*Y - 10
    return partial_x, partial_y


def bisection_method(func, a, b, tol=1e-5):
    while (b - a) / 2 > tol:
        x1 = (a + b - tol) / 2
        x2 = (a + b + tol) / 2
        if func(x1) < func(x2):
            b = x2
        else:
            a = x1
    return (a + b) / 2


def steepest_descent(x_init, y_init, alpha_init, eps=1e-5):
    x_current = x_init
    y_current = y_init
    n = 0

    while np.sqrt(gradient_J(x_current, y_current)[0] ** 2 + gradient_J(x_current, y_current)[1] ** 2) >= eps:
        grad_x, grad_y = gradient_J(x_current, y_current)
        alpha = alpha_init

        x_next = x_current - alpha * grad_x
        y_next = y_current - alpha * grad_y

        alpha_opt= bisection_method(lambda alpha: J(x_current - alpha * grad_x, y_current - alpha * grad_y), 0, alpha)

        x_current -= alpha_opt * grad_x
        y_current -= alpha_opt * grad_y

        n += 1

    return x_current, y_current, n


x_init = 1.0
y_init = -1.0
alpha_init = 0.1

x_min, y_min, iterations = steepest_descent(x_init, y_init, alpha_init)
print("Минимальное значение (x, y) = ({:.4f}, {:.4f}), J(x, y) = {:.5f}".format(x_min, y_min, J(x_min, y_min)))
print("Количество итераций:", iterations)
