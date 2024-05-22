import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Параметры нормального распределения
mu, sigma = 0, 1  # среднее и СКО

# Генерация выборки из нормального распределения
sample_size = 1000
sample = np.random.normal(mu, sigma, sample_size)

# Построение частотной гистограммы
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(sample, bins=30, density=True,
         alpha=0.6, color='b', edgecolor='black')
plt.title('Частотная гистограмма')
plt.xlabel('Значение')
plt.ylabel('Плотность')

# Добавление кривой нормального распределения на гистограмму
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
title = "Частотная гистограмма с кривой нормального распределения"
plt.title(title)

# Построение QQplot
plt.subplot(1, 2, 2)
stats.probplot(sample, dist="norm", plot=plt)
plt.title('QQplot')

plt.tight_layout()
plt.show()
