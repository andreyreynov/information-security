# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html#scipy.stats.alpha

# import matplotlib as mpl  # [m]athematics [p]lotting [l]ibrary
# from matplotlib import pyplot as plt
# from IPython.display import display, Math
# import sympy as sp
# from scipy import stats
# import numpy as np
# import random
# from IPython.display import set_matplotlib_formats

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# Задаем параметры нормального распределения
mu, sigma = 0, 1  # МО (mu) и СКО (sigma)

# Создаем массив значений от -4 до 4 с шагом 0.1
x = np.arange(-4, 4, 0.1)

# Расчет функции плотности вероятности (pdf)
pdf_values = norm.pdf(x, mu, sigma)

# Расчет кумулятивной функции (cdf)
cdf_values = norm.cdf(x, mu, sigma)

# Расчет функции точки вероятности (ppf) для вероятностей от 0 до 1
ppf_values = norm.ppf(np.linspace(0, 1, 100), mu, sigma)

# Создание графиков
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# График функции плотности вероятности (pdf)
axs[0].plot(x, pdf_values)
axs[0].set_title('Функция плотности вероятности (PDF)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Плотность')

# График кумулятивной функции (cdf)
axs[1].plot(x, cdf_values)
axs[1].set_title('Кумулятивная функция (CDF)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Вероятность')

# График функции точки вероятности (ppf)
axs[2].plot(np.linspace(0, 1, 100), ppf_values)
axs[2].set_title('Функция точки вероятности (PPF)')
axs[2].set_xlabel('Вероятность')
axs[2].set_ylabel('x')

# Отображение графиков
plt.tight_layout()
plt.show()

# Вычисление среднего, дисперсии, асимметрии и эксцесса
mean = np.mean(x)
variance = np.var(x)
skewness = skew(x)
kurt = kurtosis(x)

# Печать результатов
print(f"Среднее: {mean}")
print(f"Дисперсия: {variance}")
print(f"Асимметрия: {skewness}")
print(f"Эксцесс: {kurt}")
