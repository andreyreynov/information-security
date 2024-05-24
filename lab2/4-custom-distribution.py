import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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


# Моделирование сходимости выборочных средних для пользовательского распределения
def generate_custom_samples(amplitude, size):
    # Генерация выборок из пользовательского распределения
    return amplitude * np.sin(np.random.uniform(0, np.pi, size))


sample_size = 1000

# Генерация выборок для первого набора параметров
samples1 = generate_custom_samples(amplitude1, sample_size)

# Накопление сумм и вычисление выборочных средних
cumulative_sums = np.cumsum(samples1)
sample_means = cumulative_sums / np.arange(1, sample_size + 1)

# Построение графика сходимости выборочных средних для первого набора параметров
plt.figure(figsize=(10, 5))
plt.plot(sample_means, label=f'amplitude={amplitude1}')
plt.plot([np.mean(samples1)] * sample_size, color='k',
         linestyle='--', linewidth=1, label='Среднее значение')
plt.fill_between(np.arange(1, sample_size + 1),
                 sample_means - 0.5 / np.sqrt(np.arange(1, sample_size + 1)),
                 sample_means + 0.5 / np.sqrt(np.arange(1, sample_size + 1)),
                 color='gray', alpha=0.3)
plt.title('Сходимость выборочных средних для первого набора параметров')
plt.xlabel('n')
plt.ylabel(r'$\overline{y} = \frac{1}{n} \sum_{i=1}^n y_i$')
plt.legend()
plt.grid(True)
plt.show()

# Генерация выборок для второго набора параметров
samples2 = generate_custom_samples(amplitude2, sample_size)

# Накопление сумм и вычисление выборочных средних
cumulative_sums = np.cumsum(samples2)
sample_means = cumulative_sums / np.arange(1, sample_size + 1)

# Построение графика сходимости выборочных средних для второго набора параметров
plt.figure(figsize=(10, 5))
plt.plot(sample_means, label=f'amplitude={amplitude2}')
plt.plot([np.mean(samples2)] * sample_size, color='k',
         linestyle='--', linewidth=1, label='Среднее значение')
plt.fill_between(np.arange(1, sample_size + 1),
                 sample_means - 0.5 / np.sqrt(np.arange(1, sample_size + 1)),
                 sample_means + 0.5 / np.sqrt(np.arange(1, sample_size + 1)),
                 color='gray', alpha=0.3)
plt.title('Сходимость выборочных средних для второго набора параметров')
plt.xlabel('n')
plt.ylabel(r'$\overline{y} = \frac{1}{n} \sum_{i=1}^n y_i$')
plt.legend()
plt.grid(True)
plt.show()
