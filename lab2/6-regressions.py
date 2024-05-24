import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame
subset_data = data.head(10)
x = subset_data['MedInc'].values
y = subset_data['MedHouseVal'].values


def f(x, a, b):
    return a + b * x


'''
Подгонка по методу наименьших квадратов
'''
pars, cov_pars = curve_fit(f, x, y)
plt.plot(x, f(x, *pars), label="OLS fit")
plt.scatter(x, y, facecolor='k', label="data")
plt.legend(frameon=False)
plt.title("Подгонка по методу наименьших квадратов (OLS)")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.show()

'''
Подгонка с помощью весов
'''
sigma = np.std(y) / np.sqrt(len(y)) * np.ones_like(y)

# Подгонка с помощью весов
pars, pcov = curve_fit(f, x, y, sigma=sigma)
plt.plot(x, f(x, *pars), label="WLS fit")
plt.errorbar(x, y, yerr=sigma, fmt="ok", lw=1,
             capsize=3, markersize=5, label='data')
plt.legend(frameon=False)
plt.title("Подгонка с помощью весов (WLS)")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.show()

'''
Подгонка с учетом корреляций
'''
# Ковариационная матрица с небольшими корреляциями
ycov = np.diag(np.std(y) ** 2 / np.sqrt(len(y)) * np.ones_like(y))

pars, pcov = curve_fit(f, x, y, sigma=ycov)
plt.plot(x, f(x, *pars), label="GLS fit")
plt.errorbar(x, y, yerr=np.sqrt(np.diag(ycov)), fmt="ok",
             lw=1, capsize=3, markersize=5, label='data')
plt.legend(frameon=False)
plt.title("Подгонка с учетом корреляций (GLS)")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.show()
