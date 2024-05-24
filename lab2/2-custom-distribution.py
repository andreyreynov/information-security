import numpy as np
import matplotlib.pyplot as plt

# Функция плотности вероятности (PDF) на основе синусоиды с разной амплитудой


def custom_pdf(x, amplitude):
    return amplitude * np.sin(x)

# Кумулятивная функция распределения (CDF) на основе косинусоиды с разной амплитудой


def custom_cdf(x, amplitude):
    return (1 + amplitude * np.cos(x)) / 2


# Создаем массив значений x от 0 до pi
x_values = np.linspace(0, np.pi, 100)

# Выбираем два различных набора параметров для моделирования
amplitude1 = 0.5
amplitude2 = 1.0

# Вычисляем значения PDF и CDF для каждого набора параметров
pdf_values1 = custom_pdf(x_values, amplitude1)
cdf_values1 = custom_cdf(x_values, amplitude1)

pdf_values2 = custom_pdf(x_values, amplitude2)
cdf_values2 = custom_cdf(x_values, amplitude2)

# Создаем графики
plt.figure(figsize=(12, 6))

# График PDF для первого набора параметров
plt.subplot(1, 2, 1)
plt.plot(x_values, pdf_values1, label=f'amplitude={amplitude1}')
plt.title('Функция плотности вероятности (PDF)')
plt.xlabel('x')
plt.ylabel('Вероятность')
plt.legend()

# График CDF для первого набора параметров
plt.subplot(1, 2, 2)
plt.plot(x_values, pdf_values2, label=f'amplitude={amplitude2}')
plt.title('Функция плотности вероятности (PDF)')
plt.xlabel('x')
plt.ylabel('Вероятность')
plt.legend()

# Создаем графики для второго набора параметров
plt.figure(figsize=(12, 6))

# График PDF для второго набора параметров
plt.subplot(1, 2, 1)
plt.plot(x_values, cdf_values1, label=f'amplitude={amplitude1}')
plt.title('Кумулятивная функция (CDF)')
plt.xlabel('x')
plt.ylabel('Вероятность')
plt.legend()

# График CDF для второго набора параметров
plt.subplot(1, 2, 2)
plt.plot(x_values, cdf_values2, label=f'amplitude={amplitude2}')
plt.title('Кумулятивная функция (CDF)')
plt.xlabel('x')
plt.ylabel('Вероятность')
plt.legend()

plt.tight_layout()
plt.show()
