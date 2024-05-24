import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Загрузка датасета
california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

# Показать первые несколько строк датасета
print(data.head())

# Выбор двух признаков
features_2 = data[['MedInc', 'AveRooms']]

# Вычисление ковариационной матрицы для двух признаков
cov_matrix_2 = np.cov(features_2, rowvar=False)
print("Ковариационная матрица для двух признаков:\n", cov_matrix_2)

# Выбор трех признаков
features_3 = data[['MedInc', 'AveRooms', 'AveOccup']]

# Вычисление ковариационной матрицы для трех признаков
cov_matrix_3 = np.cov(features_3, rowvar=False)
print("Ковариационная матрица для трех признаков:\n", cov_matrix_3)
