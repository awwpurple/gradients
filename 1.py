import matplotlib.pyplot as plt
import numpy as np
import time
from numpy import linalg as ln


def method_constant_step(A, b, x, e, imax, alpha):
    """
    Функция для постоянного шага альфа
    a - альфа - число типа float, шаг, на который мы изменяем значение невязки
    для приближения текущих х к истинным значениям,
    A - матрица коэффициентов,
    b - столбец свободных членов,
    x - столбец предполагаемых решений,
    e - допустимая погрешность,
    imax - максимальное количество итераций
    """
    i = 0  # счетчик итераций
    residual = b - A.dot(x)  # residual - невязка, столбец (массив) со значениями типа float
    d = np.transpose(residual).dot(residual)  # d - квадрат значения невязки, число типа float
    d0 = d  # d0 - начальное значение d, число float
    while (i < imax) and (d > e ** 2 * d0):  # цикл, ограниченный максимальным кол-вом итераций и погрешностью
        x = x + alpha * residual  # заново высчитывается более приближенное значение столбца х с помощью альфа и невязки
        residual = b - A.dot(x)  # пересчитывается значение невязки с учетом измененных решений
        d = np.transpose(residual).dot(residual)  # дельта с измененной невязкой
        i += 1
    return x, np.sqrt(d), i  # возвращаем приближенные решения, значение ошибки, кол-во итераций


def method_variable_step(A, b, x, e, imax):
    """
    Функция для переменного шага альфа
    a - альфа - число типа float, шаг, на который мы изменяем значение невязки
    для приближения текущих х к истинным значениям,
    A - матрица коэффициентов,
    b - столбец свободных членов,
    x - столбец предполагаемых решений,
    e - допустимая погрешность,
    imax - максимальное количество итераций
    """
    x_array = np.empty((0, len(x)), float)
    i = 0  # счетчик итераций
    residual = b - A.dot(x)  # residual - невязка, столбец (массив) со значениями типа float
    d = np.transpose(residual).dot(residual)  # d - квадрат значения невязки, число типа float
    d0 = d  # d0 - начальное значение d, число float
    while (i < imax) and (d > e ** 2 * d0):  # цикл, ограниченный максимальным кол-вом итераций и погрешностью
        q = A.dot(residual)  # умножается матрица коэффициентов на невязку
        alpha = d / (np.transpose(residual).dot(q))  # высчитывается альфа
        x = x + alpha * residual  # заново высчитывается более приближенное значение столбца х с помощью альфа и невязки
        residual = b - A.dot(x)  # пересчитывается значение невязки с учетом измененных решений
        d = np.transpose(residual).dot(residual)  # дельта с измененной невязкой
        i += 1
        x_array = np.append(x_array, np.array([x]), axis=0)
    return x, np.sqrt(d), i, x_array  # Возвращаем полученные значения х, невязку и кол-во произведенных итераций


def steepest_decent_examples(matrix_a2, b2, x2, e2, imax2, alpha2, matrix_a3, b3, x3, e3, imax3, alpha3, matrix_a4, b4,
                             x4, e4, imax4, alpha4):
    print("Matrix 2x2")

    print("Method_constant_step")
    print(method_constant_step(matrix_a2, b2, x2, e2, imax2, alpha2))
    print("Method_variable_step")
    print(method_variable_step(matrix_a2, b2, x2, e2, imax2))

    print("Matrix 3x3")
    print("Method_constant_step")
    print(method_constant_step(matrix_a3, b3, x3, e3, imax3, alpha3))
    print("Method_variable_step")
    print(method_variable_step(matrix_a3, b3, x3, e3, imax3))

    print("Matrix 4x4")
    print("Method_constant_step")
    print(method_constant_step(matrix_a4, b4, x4, e4, imax4, alpha4))
    print("Method_variable_step")
    print(method_variable_step(matrix_a4, b4, x4, e4, imax4))


def graphics_to_steepest_decent(matrix_a2, b2, x2, e2, imax2, matrix_a3, b3, x3, e3, imax3, matrix_a4, b4, x4, e4,
                                imax4):
    j = 0
    alphas = np.arange(0, 1, 0.05)
    d_array = np.zeros(len(alphas))
    for alpha in alphas:
        x, d, i = method_constant_step(matrix_a2, b2, x2, e2, imax2, alpha)
        d_array[j] = d
        j += 1
    alpha_formula = np.transpose(b2 - matrix_a2.dot(x2)).dot(b2 - matrix_a2.dot(x2)) / (
        np.transpose(b2 - matrix_a2.dot(x2)).dot(matrix_a2.dot(b2 - matrix_a2.dot(x2))))
    fig, ax = plt.subplots()
    plt.title("d(a) for 2x2 matrix")
    plt.xlabel("alpha")
    plt.ylabel("d")
    plt.yscale('log')
    ax.vlines(alpha_formula, 0, d_array.max(), color='black')
    ax.plot(alphas, d_array, color='black')
    ax.grid()
    plt.show()

    j = 0
    alphas2 = np.arange(0.05, 0.4, 0.01)
    d_array2 = np.zeros(len(alphas2))
    for alpha2 in alphas2:
        x, d, i = method_constant_step(matrix_a2, b2, x2, e2, imax2, alpha2)
        d_array2[j] = d
        j += 1
    fig, ax = plt.subplots()
    plt.title("enlarged fragment d(a) for 2x2 matrix")
    plt.xlabel("alpha")
    plt.ylabel("d")
    plt.yscale('log')
    ax.vlines(alpha_formula, 0, d_array2.max(), color='black')
    ax.plot(alphas2, d_array2, color='black')
    ax.grid()
    plt.show()

    j = 0
    alphas = np.arange(0, 1, 0.05)
    d_array = np.zeros(len(alphas))
    for alpha in alphas:
        x, d, i = method_constant_step(matrix_a3, b3, x3, e3, imax3, alpha)
        d_array[j] = d
        j += 1
    alpha_formula = np.transpose(b3 - matrix_a3.dot(x3)).dot(b3 - matrix_a3.dot(x3)) / (
        np.transpose(b3 - matrix_a3.dot(x3)).dot(matrix_a3.dot(b3 - matrix_a3.dot(x3))))
    fig, ax = plt.subplots()
    plt.title("d(a) for 3x3 matrix")
    plt.xlabel("alpha")
    plt.ylabel("d")
    plt.yscale('log')
    ax.plot(alphas, d_array, color='black')
    ax.vlines(alpha_formula, 0, d_array.max(), color='black')
    ax.grid()
    plt.show()

    j = 0
    alphas2 = np.arange(0, 0.23, 0.01)
    d_array2 = np.zeros(len(alphas2))
    for alpha2 in alphas2:
        x, d, i = method_constant_step(matrix_a3, b3, x3, e3, imax3, alpha2)
        d_array2[j] = d
        j += 1
    fig, ax = plt.subplots()
    plt.title("enlarged fragment d(a) for 3x3 matrix")
    plt.xlabel("alpha")
    plt.ylabel("d")
    plt.yscale('log')
    ax.vlines(alpha_formula, 0, d_array2.max(), color='black')
    ax.plot(alphas2, d_array2, color='black')
    ax.grid()
    plt.show()

    j = 0
    alphas = np.arange(0, 1, 0.05)
    d_array = np.zeros(len(alphas))
    for alpha in alphas:
        x, d, i = method_constant_step(matrix_a4, b4, x4, e4, imax4, alpha)
        d_array[j] = d
        j += 1
    alpha_formula = np.transpose(b4 - matrix_a4.dot(x4)).dot(b4 - matrix_a4.dot(x4)) / (
        np.transpose(b4 - matrix_a4.dot(x4)).dot(matrix_a4.dot(b4 - matrix_a4.dot(x4))))
    fig, ax = plt.subplots()
    plt.title("d(a) for 4x4 matrix")
    plt.xlabel("alpha")
    plt.ylabel("d")
    plt.yscale('log')
    ax.plot(alphas, d_array, color='black')
    ax.vlines(alpha_formula, 0, d_array.max(), color='black')
    ax.grid()
    plt.show()

    j = 0
    alphas2 = np.arange(0.7, 0.95, 0.01)
    d_array2 = np.zeros(len(alphas2))
    for alpha2 in alphas2:
        x, d, i = method_constant_step(matrix_a4, b4, x4, e4, imax4, alpha2)
        d_array2[j] = d
        j += 1
    fig, ax = plt.subplots()
    plt.title("enlarged fragment d(a) for 4x4 matrix")
    plt.xlabel("alpha")
    plt.ylabel("d")
    plt.yscale('log')
    ax.vlines(alpha_formula, 0, d_array2.max(), color='black')
    ax.plot(alphas2, d_array2, color='black')
    ax.grid()
    plt.show()


def conjugate_gradients(A, b, x, e, imax):
    x_array = np.empty((0, 2), float)
    i = 0  # number
    residual = b - A.dot(x)  # array
    d = residual  # array
    delta_new = np.transpose(residual).dot(residual)  # number
    delta0 = delta_new  # number
    while i < imax and delta_new > e ** 2 * delta0:
        q = A.dot(d)  # array
        alpha = delta_new / (np.transpose(d).dot(q))  # number
        x = x + alpha * d  # array
        residual = b - A.dot(x)  # array
        delta_old = delta_new
        delta_new = np.transpose(residual).dot(residual)
        betta = delta_new / delta_old
        d = residual + betta * d
        i += 1
        x_array = np.append(x_array, np.array([x]), axis=0)
    return x, delta_new, i, x_array


def method_polak_ribeire(A, b, x, imax, e_cg, sigma0, jmax, e_s):
    i = 0
    k = 0
    x_array = np.empty((0, len(x)), float)
    residual = -1 * (A.dot(x) - b)  # вектор
    m = A  # матрица Гессе
    s = ln.matrix_power(m, -1).dot(residual)  # вектор
    d = s  # вектор
    delta_new = np.transpose(residual).dot(d)  # число
    n = 2
    delta0 = delta_new  # число
    while i < imax and delta_new > e_cg ** 2 * delta0:
        j = 0
        delta_d = np.transpose(d).dot(d)  # число
        alpha = - sigma0
        eta_prev = np.transpose(A.dot(x + sigma0 * d) - b).dot(d)  # число
        while (j < jmax) and (alpha ** 2 * delta_d) > e_s ** 2:
            eta = np.transpose(A.dot(x) - b).dot(d)  # число
            alpha = alpha * (eta / (eta_prev - eta))  # число
            x = x + alpha * d
            x_array = np.append(x_array, np.array([x]), axis=0)
            eta_prev = eta
            j = j + 1
            if (j >= jmax) and (alpha ** 2 * delta_d) >= e_s ** 2:
                break

        residual = -1 * (A.dot(x) - b)
        delta_old = delta_new
        delta_mid = np.transpose(residual).dot(s)
        m = A
        s = ln.matrix_power(m, -1).dot(residual)
        delta_new = np.transpose(residual).dot(s)
        betta = (delta_new - delta_mid) / delta_old
        k = k + 1
        if k == n or betta <= 0:
            d = s
            k = 0
        else:
            d = s + betta * d
        i = i + 1
    return x, np.sqrt(delta_new), i, x_array


def conjugrate_gradients_example(matrix_a, b, x, imax, e, c):
    fig, ax = plt.subplots()
    print("Method of conjugate gradients")
    start_time_gradients = time.time()
    approximated_x_gradients, d_gradients, i_gradients, x_array_gradients = conjugate_gradients(matrix_a, b, x, e, imax)
    print("--- running time %s seconds ---" % (time.time() - start_time_gradients))
    print("Approximate x values: ", approximated_x_gradients, ", norm value: ", d_gradients, ", number of iterations: ",
          i_gradients)
    print("All approximations:", x_array_gradients)
    x_graph_gradients = np.array([x[0]], dtype=float)
    y_graph_gradients = np.array([x[1]], dtype=float)
    for coordinates in x_array_gradients:
        x_graph_gradients = np.append(x_graph_gradients, coordinates[0])
        y_graph_gradients = np.append(y_graph_gradients, coordinates[1])
    ax.plot(x_graph_gradients, y_graph_gradients, marker='o', linestyle='--', color='black', alpha=0.5)
    x1 = np.arange(-6, 6.5, 0.5)
    x2 = np.arange(-6, 6.5, 0.5)
    i, j = 0, 0
    z = np.zeros((len(x1), len(x1)))
    for x1_curr in x1:
        for x2_curr in x2:
            x0 = np.array([x1_curr, x2_curr])
            z[i, j] = 0.5 * np.transpose(x0).dot(matrix_a).dot(x0) - np.transpose(b).dot(x0) + c
            i += 1
        j += 1
        i = 0

    lev = np.arange(1, 50, 5)
    ax.contour(x1, x2, z, levels=lev, colors='gray')
    ax.grid()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def visualisation_pr_sd(matrix_a, b, x, imax, e, sigma0, jmax, c):
    f2 = open('output', 'w')
    fig, ax = plt.subplots()
    f2.write("Polak Ribier method\n")
    start_time_gradients = time.time()
    approximated_x_pr, d_pr, i_pr, x_array_pr = method_polak_ribeire(matrix_a, b, x, imax, e, sigma0, jmax, e)
    f2.write("--- running time %s seconds ---" % (time.time() - start_time_gradients))
    f2.write("\nApproximate x values: ")
    f2.write(str(approximated_x_pr))
    f2.write("\nnorm value: ")
    f2.write(str(d_pr))
    f2.write("\nnumber of iterations: ")
    f2.write(str(i_pr))
    f2.write("\nAll approximations: ")
    for line in x_array_pr:
        f2.write(str(line))
        f2.write("\n")
    if len(matrix_a) == 2:
        x_graph_gradients = np.array([x[0]], dtype=float)
        y_graph_gradients = np.array([x[1]], dtype=float)
        for coordinates1 in x_array_pr:
            x_graph_gradients = np.append(x_graph_gradients, coordinates1[0])
            y_graph_gradients = np.append(y_graph_gradients, coordinates1[1])
        plt.plot(x_graph_gradients, y_graph_gradients, marker='o', linestyle='--', color='black', alpha=0.5)
    f2.write("Method of steepest decent")
    start_time_steepest = time.time()
    approximated_x_steepest, d_steepest, i_steepest, x_array_steepest = method_variable_step(matrix_a, b, x, e, imax)
    f2.write("--- running time %s seconds ---" % (time.time() - start_time_steepest))
    f2.write("\nApproximate x values: ")
    f2.write(str(approximated_x_steepest))
    f2.write("\nnorm value: ")
    f2.write(str(d_steepest))
    f2.write("\nnumber of iterations: ")
    f2.write(str(i_steepest))
    f2.write("\nAll approximations: ")
    for line in x_array_steepest:
        f2.write(str(line))
        f2.write("\n")
    if len(matrix_a):
        x_graph_steepest = np.array([x[0]], dtype=float)
        y_graph_steepest = np.array([x[1]], dtype=float)
        for coordinates2 in x_array_steepest:
            x_graph_steepest = np.append(x_graph_steepest, coordinates2[0])
            y_graph_steepest = np.append(y_graph_steepest, coordinates2[1])
        plt.plot(x_graph_steepest, y_graph_steepest, marker='^', color='black', alpha=0.5)

    x1 = np.arange(-6, 6.5, 0.5)
    x2 = np.arange(-6, 6.5, 0.5)
    i, j = 0, 0
    z = np.zeros((len(x1), len(x1)))
    if len(matrix_a) == 2:
        for x1_curr in x1:
            for x2_curr in x2:
                x0 = np.array([x1_curr, x2_curr])
                z[i, j] = 0.5 * np.transpose(x0).dot(matrix_a).dot(x0) - np.transpose(b).dot(x0) + c
                i += 1
            j += 1
            i = 0

        lev = np.arange(1, 50, 5)
        ax.contour(x1, x2, z, levels=lev, colors='gray')
        ax.grid()
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
    f2.close()


def main():
    matrix_a2 = np.array([[2, 3], [3, 9]], dtype=float)
    b2 = np.transpose(np.array([6, 14], dtype=float))
    x2 = np.transpose(np.array([-2, 3], dtype=float))
    e2 = 0.01
    imax2 = 100
    alpha2 = 0.13
    # x = (1.5, 1)
    matrix_a3 = np.array([[1, 2, 3], [3, 5, 7], [1, 3, 4]], dtype=float)
    b3 = np.transpose(np.array([3, 0, 1], dtype=float))
    x3 = np.transpose(np.array([-3, -12, 10], dtype=float))
    e3 = 0.15
    imax3 = 100
    alpha3 = 0.1
    # x = (-4, -13, 11)
    matrix_a4 = np.array([[1, -1, 3, 1], [4, -1, 5, 4], [2, -2, 4, 1], [1, -4, 5, -1]], dtype=float)
    # matrix_a4 = np.eye(4) + np.diag(np.ones(3), k=1)
    b4 = np.transpose(np.array([5, 4, 6, 3], dtype=float))
    x4 = np.transpose(np.array([7, 15, 11, -15], dtype=float))
    e4 = 0.1
    imax4 = 100
    alpha4 = 0.9
    # x = (4, 1, 3, 3)

    steepest_decent_examples(matrix_a2, b2, x2, e2, imax2, alpha2, matrix_a3, b3, x3, e3, imax3, alpha3, matrix_a4, b4, x4, e4, imax4, alpha4)
    graphics_to_steepest_decent(matrix_a2, b2, x2, e2, imax2, matrix_a3, b3, x3, e3, imax3, matrix_a4, b4, x4, e4, imax4)
    matrix_a_cg = np.array([[2, 3], [3, 9]], dtype=float)
    b_cg = np.transpose(np.array([6, 14], dtype=float))
    x_cg = np.transpose(np.array([-2, 3], dtype=float))
    imax_cg = 100
    e_cg = 0.01
    c_cg = 0
    # x =[2, -2]
    conjugrate_gradients_example(matrix_a_cg, b_cg, x_cg, imax_cg, e_cg, c_cg)

    f = open('input', 'r')
    matrix_a = np.array(f.readline().rstrip().split(), dtype=float)
    n = len(matrix_a)
    matrix_a = matrix_a.reshape((int(np.sqrt(n)), int(np.sqrt(n))))
    b = np.transpose(np.array(f.readline().rstrip().split(), dtype=float))
    x = np.transpose(np.array(f.readline().rstrip().split(), dtype=float))
    imax = int(f.readline().rstrip())
    e = float(f.readline().rstrip())
    sigma0 = float(f.readline().rstrip())
    jmax = int(f.readline().rstrip())
    c = float(f.readline().rstrip())
    # x =[2, -2]
    visualisation_pr_sd(matrix_a, b, x, imax, e, sigma0, jmax, c)
    f.close()


if __name__ == "__main__":
    main()
