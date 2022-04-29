import numpy as np


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
    return x, d, i  # возвращаем приближенные решения, значение ошибки, кол-во итераций


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
    return x, d, i  # Возвращаем полученные значения х, невязку и кол-во произведенных итераций


def main():
    print("Matrix 2x2")
    matrix_a = np.array([[2, 3], [4, 9]], dtype=float)
    b = np.transpose(np.array([6, 15], dtype=float))
    x = np.transpose(np.array([1, 1], dtype=float))
    e = 0.1
    imax = 100
    alpha = 0.9
    # x = (1.5, 1)
    print("Method_constant_step")
    print(method_constant_step(matrix_a, b, x, e, imax, alpha))
    print("Method_variable_step")
    print(method_variable_step(matrix_a, b, x, e, imax))

    print("Matrix 3x3")
    matrix_a = np.array([[1, 2, 3], [3, 5, 7], [1, 3, 4]], dtype=float)
    b = np.transpose(np.array([3, 0, 1], dtype=float))
    x = np.transpose(np.array([-3, -12, 10], dtype=float))
    e = 0.1
    imax = 10000
    alpha = 0.1
    # x = (-4, -13, 11)
    print("Method_constant_step")
    print(method_constant_step(matrix_a, b, x, e, imax, alpha))
    print("Method_variable_step")
    print(method_variable_step(matrix_a, b, x, e, imax))

    print("Matrix 4x4")
    matrix_a = np.array([[1, -1, 3, 1], [4, -1, 5, 4], [2, -2, 4, 1], [1, -4, 5, -1]], dtype=float)
    b = np.transpose(np.array([5, 4, 6, 3], dtype=float))
    x = np.transpose(np.array([7, 15, 11, -15], dtype=float))
    e = 0.1
    imax = 100
    alpha = 0.09
    # x = (9, 18, 10, -16)
    print("Method_constant_step")
    print(method_constant_step(matrix_a, b, x, e, imax, alpha))
    print("Method_variable_step")
    print(method_variable_step(matrix_a, b, x, e, imax))


if __name__ == "__main__":
    main()
