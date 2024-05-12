import numpy as np


# Определение целевой функции
def function_to_minimize(X, Y):
    return 7*(X**2) + 2*X*Y + 5*(Y**2) + X - 10*Y


# Метод Нелдера-Мида для оптимизации
def nelder_mead(f, x0, alpha, beta, gamma, eps):
    count = 0
    x = np.array(x0)
    n = len(x)
    X = [x]

    # Инициализация симплекса
    for i in range(n):
        x_new = x.copy()
        x_new[i] += 1
        X.append(x_new)

    while np.linalg.norm(X[0] - X[-1]) >= eps:
        X.sort(key=lambda x: f(*x))  # Сортировка точек по значению функции

        x_c = np.mean(X[:-1], axis=0)  # Центр масс симплекса

        # Отражение
        x_r = x_c + alpha * (x_c - X[-1])

        if f(*x_r) < f(*X[0]):
            # Если отраженная точка лучше лучшей точки, заменяем ей худшую точку
            x_e = x_c + gamma * (x_r - x_c)
            if f(*x_e) < f(*x_r):
                X[-1] = x_e
            else:
                X[-1] = x_r
        else:
            # Операции растяжения, сжатия и уменьшения
            x_s = x_c + beta * (X[-1] - x_c)
            if f(*x_s) < f(*X[-1]):
                X[-1] = x_s
            else:
                for i in range(1, len(X)):
                    X[i] = 0.5 * (X[i] - X[0]) + X[0]

        count += 1

    return X[0], f(*X[0]), count


# Параметры и начальное приближение
x0 = [1, 1]
eps = 1e-6
alpha = 1.0
beta = 0.5
gamma = 2.0

# Вызов метода Нелдера-Мида для оптимизации
x, F_min, count = nelder_mead(function_to_minimize, x0, alpha, beta, gamma, eps)

# Вывод результатов
print("Точка минимума (X, Y):", x[0], x[1])
print("Минимальное значение функции:", F_min)
print("Iterations:", count)
