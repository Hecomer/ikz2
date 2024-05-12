import numpy as np


def function_to_minimize(X, Y, Z):
    return 3*(X**2) + 4*(Y**2) + 5*(Z**2) + 2*X*Y - X*Z-2*Y*Z + X - 3*Z


def nelder_mead(f, x0, alpha, beta, gamma, eps):
    count = 0
    x = np.array(x0)
    n = len(x)
    X = [x]

    for i in range(n):
        x_new = x.copy()
        x_new[i] += 1
        X.append(x_new)

    while np.linalg.norm(X[0] - X[-1]) >= eps:
        #сортируем по значению функции
        X.sort(key=lambda x: f(*x))
        #centr mass
        x_c = np.mean(X[:-1], axis=0)
        #otrazhennaya otnositelno centra
        x_r = x_c + alpha * (x_c - X[-1])
        X[-1] = x_r
        #obnovlenie simplexa otnosit
        if f(*x_r) < f(*X[0]):
            #esli otrazhennaya tochka luche luchshey, to zamenyaem hudshuyu
            x_e = x_c + gamma * (x_r - x_c)
            if f(*x_e) < f(*x_r):
                X[-1] = x_e
            else:
                X[-1] = x_r
        else:
            #rastyazhenia szhatia
            x_s = x_c + beta * (X[-1] - x_c)
            if f(*x_s) < f(*X[-1]):
                X[-1] = x_s
            else:
                for i in range(1, len(X)):
                    X[i] = 0.5 * (X[i] - X[0]) + X[0]
        count += 1
    return X[0], f(*X[0]), count


x0 = [1, 1, 1]
eps = 1e-6
alpha = 1.0
beta = 0.5
gamma = 2.0

x, F_min, count = nelder_mead(function_to_minimize, x0, alpha, beta, gamma, eps)


print("Точка минимума (X, Y, Z):", x[0], x[1], x[2])
print("Минимальное значение функции:", F_min)
print("Iterations:", count)
