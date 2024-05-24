# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html#scipy.stats.bernoulli

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
from scipy.stats import bernoulli, skew, kurtosis

# Задаем вероятность успеха (p) в распределении Бернулли
p = 0.5

# Создаем распределение Бернулли с вероятностью успеха p
bernoulli_dist = bernoulli(p)

# Создаем массив значений 0 и 1
x = np.array([0, 1])

# Вычисляем функцию плотности вероятности (PMF)
pmf_values = bernoulli_dist.pmf(x)

# Вычисляем кумулятивную функцию (CDF)
cdf_values = bernoulli_dist.cdf(x)

# Вычисляем функцию точки вероятности (PPF) для вероятностей от 0 до 1
ppf_values = bernoulli_dist.ppf(np.linspace(0, 1, 100))

# Создаем выборку из распределения Бернулли
sample_size = 1000
sample = bernoulli_dist.rvs(size=sample_size)

# Создание графиков
fig, axs = plt.subplots(4, 1, figsize=(10, 20))

# График функции плотности вероятности (PMF)
axs[0].stem(x, pmf_values, basefmt=" ")
axs[0].set_title('Функция плотности вероятности (PMF)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('Вероятность')

# График кумулятивной функции (CDF)
axs[1].step(x, cdf_values, where='post')
axs[1].set_title('Кумулятивная функция (CDF)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Вероятность')

# График функции точки вероятности (PPF)
axs[2].plot(np.linspace(0, 1, 100), ppf_values)
axs[2].set_title('Функция точки вероятности (PPF)')
axs[2].set_xlabel('Вероятность')
axs[2].set_ylabel('x')

# Построение гистограммы выборки
axs[3].hist(sample, bins=[-0.5, 0.5, 1.5], rwidth=0.5, color='blue', alpha=0.7)
axs[3].set_title('Гистограмма выборки')
axs[3].set_xlabel('Значение')
axs[3].set_ylabel('Частота')

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
