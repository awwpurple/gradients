def constant_step(A, b, x, e, imax):
    """
    Функция для постоянного шага альфа
    a - альфа - задается с клавиатуры, число типа float
    Альфа - шаг, на который мы изменяем значение невязки для приближения текущих х к истинным значениям
    i - номер итерации, число типа int
    A, b, x, e, imax - параметры, вводимые программно для примера: матрица коэффициентов,
    столбец свободных членов, столбец предполагаемых решений, допустимая погрешность,
    максимальное количество итераций
    r - невязка, столбец (массив) со значениями типа float
    d - квадрат значения невязки (?), число типа float
    d0 - начальное значение d, число float
    Высчитывается значение невязки,значение дельты. В цикле на каждом шаге заново высчитывается более
    приближенное значение столбца х с помощью альфа и невязки.
    Пересчитываем значение невязки r с учетом измененных решений и дельта с измененной невязкой.
    Выводим на экран полученные значения х, невязку и кол-во произведенных итераций
    """
    print("Please enter alpha")
    a = float(input())
    i = 0
    r = b - A.dot(x)
    d = np.transpose(r).dot(r)
    d0 = d
    while (i < imax) and (d > e**2 * d0):
        x = x + a * r
        r = b - A.dot(x)
        d = np.transpose(r).dot(r)
        i += 1
    print(x, r, i)


def variable_step(A, b, x, e, imax):
    """
    Функция для переменного шага альфа
    a - альфа - число типа float, изменяется в ходе программы и зависит от невязки
    Остальные параметры те же, что и в функции выше
    Алгоритм из параграфа steepest decent
    Выводим на экран полученные значения х, невязку и кол-во произведенных итераций
    """
    i = 0
    r = b - A.dot(x)
    d = np.transpose(r).dot(r)
    d0 = d
    while (i < imax) and (d > e**2 * d0):
        q = A.dot(r)
        a = d / (np.transpose(r).dot(q))
        x = x + a * r
        r = b - A.dot(x)
        d = np.transpose(r).dot(r)
        i += 1
    print(x, r, i)


import numpy as np
print("Enter '2' for 2x2 matrix, '3' for 3x3 and '4' for 4x4")
matrix = int(input())
if matrix == 2:
    A = np.array([[2, 3], [4, 9]], dtype=float)
    b = np.transpose(np.array([6, 15], dtype=float))
    x = np.transpose(np.array([1, 1], dtype=float))
    e = 0.01
    imax = 100
    #x = (1.5, 1)
elif matrix == 3:
    A = np.array([[1, 2, 3], [3, 5, 7], [1, 3, 4]], dtype=float)
    b = np.transpose(np.array([3, 0, 1], dtype=float))
    x = np.transpose(np.array([-3, -12, 10], dtype=float))
    e = 0.1
    imax = 10000
    #x = (-4, -13, 11)
elif matrix == 4:
    A = np.array([[1, -1, 3, 1], [4, -1, 5, 4], [2, -2, 4, 1], [1, -4, 5, -1]], dtype=float)
    b = np.transpose(np.array([5, 4, 6, 3], dtype=float))
    x = np.transpose(np.array([7, 15, 11, -15], dtype=float))
    e = 0.1
    imax = 100
    #x = (9, 18, 10, -16)
else:
    print("You entered an invalid value")
if matrix == 2 or matrix == 3 or matrix == 4:
    print("Enter 1 for constant step alpha and 0 for variable")
    choice = int(input())
    if choice == 1:
        constant_step(A, b, x, e, imax)
    elif choice == 0:
        variable_step(A, b, x, e, imax)
    else:
        print("You entered an invalid value")