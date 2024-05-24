import numpy as np

my_column_names = ['Lead Guitar', 'Rhythm Guitar', 'Bass', 'Drums', 'Vocals']
my_data = np.random.randint(low=0, high=5, size=(10, 5))


def cycleSum(data, column_names):
    print("Исходные данные:")
    print(data)
    print("\nЦиклические суммы значений колонок:")

    for col_idx, col_name in enumerate(column_names):
        cyclic_sum = np.roll(data[:, col_idx], 1).cumsum()
        print(f"{col_name}: {cyclic_sum}")


def getMatrixValue(matrix, x, y):
    try:
        value = matrix[x, y]
        print(f"Значение по координатам [{x}][{y}]: {value}")
    except IndexError:
        print("Ошибка! Нет такого значения.")


def multiplyColumns(column): return np.prod(column)


def sumCells(matrix_cell1, matrix_cell2):
    try:
        value_sum = matrix_cell1 + matrix_cell2
        return value_sum
    except IndexError:
        return "Ошибка! Нет такой ячейки в массиве"


print(my_data)

# Цикличекое суммирование
# cycleSum(my_data, my_column_names)


# Вывод ошибки
# getMatrixValue(my_data, 3, 2)
# getMatrixValue(my_data, 15, 2)


# Lambda-функция
# result = np.apply_along_axis(multiplyColumns, axis=0, arr=my_data)
# print(result)

# Сумма двух ячеек
# result = sumCells(my_data[2, 1], my_data[2, 4])
# print(f"Сумма значений в ячейках: {result}")
