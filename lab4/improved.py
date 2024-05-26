import os
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

warnings.filterwarnings('ignore')

# Загрузка датасета
url = 'https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv'
data = pd.read_csv(url, sep=",")

# Дополняем время к колонке Date для получения формата YYYY-MM-DD HH:MM:SS
data['Date'] = pd.to_datetime(data['Date'] + ' 01:00:00')
data.set_index('Date', inplace=True)


def pair_plot(data=data):
    # Построение попарного графика
    sns.pairplot(data)
    plt.show()


def line_plot(data=data):
    # Построение линейного графика потребления
    sns.lineplot(data['Consumption'])
    plt.title('Линейный график потребления')
    plt.show()


def autocor_plot(data=data):
    # Построение автокорреляционного графика
    plt.figure(figsize=(11, 4), dpi=80)
    pd.plotting.autocorrelation_plot(data['2012-01':'2013-01']['Consumption'])
    plt.title('Автокорреляционный график потребления')
    plt.show()


def regression_models_comparison(data=data):

    # Изменение данных
    data_consumption = data[['Consumption']].copy()
    data_consumption.loc[:, 'Yesterday'] = data_consumption.loc[:,
                                                                'Consumption'].shift()
    data_consumption.loc[:,
                         'Yesterday_Diff'] = data_consumption.loc[:, 'Yesterday'].diff()
    data_consumption = data_consumption.dropna()

    # Объявление колонок, которые будут использоваться
    X_train = data_consumption[:'2016'].drop(['Consumption'], axis=1)
    y_train = data_consumption.loc[:'2016', 'Consumption']
    X_test = data_consumption['2017-01':'2017-12'].drop(
        ['Consumption'], axis=1)
    y_test = data_consumption.loc['2017-01':'2017-12', 'Consumption']

    # Модели
    models = []
    models.append(('Модель LR', LinearRegression()))
    models.append(('Модель NN', MLPRegressor(solver='lbfgs')))
    models.append(('Модель KNN', KNeighborsRegressor()))
    models.append(('Модель RF', RandomForestRegressor(n_estimators=50)))
    models.append(('Модель SVR', SVR(gamma='auto', kernel='rbf')))
    # New models
    models.append(('Модель GB', GradientBoostingRegressor(n_estimators=50)))
    results = []
    names = []

    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=10)
        cv_results = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    # Построение графика сравнения моделей
    plt.boxplot(results, labels=names)
    plt.title('Сравнение точности моделей')
    plt.show()

    # Вывод R2, MAE, MSE, RMSE по каждой модели
    def regression_results(y_true, y_pred, model):
        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)

        print('-------------------------------------------------------')
        print(model)
        print('r2: ', round(r2, 4))
        print('MAE: ', round(mean_absolute_error, 4))
        print('MSE: ', round(mse, 4))
        print('RMSE: ', round(np.sqrt(mse), 4))
        print('-------------------------------------------------------')

    # Сравнение моделей с реальными данными
    for name, model in models:
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        regression_results(y_test, y_pred, model)

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(X_test.index, y_pred, linewidth=2, label='Предсказание')
        ax.plot(X_test.index, y_test, linewidth=2, label='Реальные данные')
        ax.set_title(f'Сравнение модели {model} с реальными данными')
        ax.legend()
        ax.set_xlabel('Year')
        ax.set_ylabel('Consumption (GWh)')
        plt.show()


def basic_data_smoothing_and_visualization(data=data):

    def prepare_data(data):
        # Создание папок
        os.makedirs('temp/basic_data_smoothing', exist_ok=True)

        # Редактирование данных
        data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
        data_7d_rol = data[data_columns].rolling(window=7, center=True).mean()
        data_365d_rol = data[data_columns].rolling(
            window=365, center=True).mean()
        data_365d_rol['Consumption'] = data_365d_rol['Consumption'].bfill().ffill()

        # Сохранение переменных с помощью pickle
        with open('temp/basic_data_smoothing/basic_data_smoothing.pkl', 'wb') as f:
            pickle.dump((data_columns, data_7d_rol, data_365d_rol), f)

    prepare_data(data)
    with open('temp/basic_data_smoothing/basic_data_smoothing.pkl', 'rb') as f:
        data_columns, data_7d_rol, data_365d_rol = pickle.load(f)

    # Настройки графика
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data['Consumption'], marker='.', markersize=2,
            color='0.6', linestyle='None', label='Daily')
    ax.plot(data_7d_rol['Consumption'], linewidth=2, label='7-d Rolling Mean')
    ax.plot(data_365d_rol['Consumption'], color='0.2',
            linewidth=3, label='Trend (365-d Rolling Mean)')
    ax.legend()
    ax.set_xlabel('Years')
    ax.set_ylabel('Consumption (GWh)')
    ax.set_title('Trends in Electricity Consumption')
    plt.show()


def enhanced_data_corrections_and_visualization(data=data):

    def prepare_data():
        os.makedirs('temp/enchanced_data_corrections', exist_ok=True)

        with open('temp/basic_data_smoothing/basic_data_smoothing.pkl', 'rb') as f:
            data_columns, data_7d_rol, data_365d_rol = pickle.load(f)

        data_correction = data[data_columns].copy()
        data_correction['trend_residuals'] = data_365d_rol['Consumption'].max(
        ) - data_365d_rol['Consumption']
        data_correction['Consumption_eq'] = data_correction['Consumption'] + \
            data_correction['trend_residuals']
        data_correction['trend_eq'] = data_365d_rol['Consumption'] + \
            data_correction['trend_residuals'] * 2
        data_correction['Consumption_eq_roll_7_day'] = data_correction['Consumption_eq'].rolling(
            window=7, center=True).mean()
        data_correction['constant'] = 1600

        # Сохранение датафрейма в формате parquet с помощью pandas
        data_correction.to_parquet(
            'temp/enchanced_data_corrections/data_correction.parquet')

        # Сохранение переменных с помощью pickle
        with open('temp/enchanced_data_corrections/enchanced_data_corrections.pkl', 'wb') as f:
            pickle.dump((data_correction, data_columns,
                        data_7d_rol, data_365d_rol), f)

    prepare_data()
    with open('temp/enchanced_data_corrections/enchanced_data_corrections.pkl', 'rb') as f:
        data_correction, data_columns, data_7d_rol, data_365d_rol = pickle.load(
            f)

    # Настройки графика
    fig, ax = plt.subplots(figsize=(28, 10))
    ax.plot(data_correction['Consumption'], marker='.',
            markersize=3, linestyle='None', label='Consumption')
    ax.plot(data_correction['Consumption_eq'], marker='.', markersize=2,
            color='0', linestyle='None', label='Consumption equalized')
    ax.plot(data_correction['Consumption_eq_roll_7_day'],
            linewidth=2, label='Trend (7-d Rolling Mean) equalized')
    ax.plot(data_7d_rol['Consumption'], linewidth=2,
            label='Trend (7-d Rolling Mean)')
    ax.plot(data_correction['trend_eq'], color='0.2',
            linewidth=3, label='Trend (365-d Rolling Mean) equalized')
    ax.plot(data_correction['constant'], linewidth=2, label='constant')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Consumption (GWh)')

    aver_trends = pd.DataFrame(columns=['data_7d_rol__mean', 'data_7d_rol__var',
                                        'data_7d_rol_eq__mean', 'data_7d_rol_eq__var'], index=data_7d_rol.index.year.unique())

    for year in data_7d_rol.index.year.unique():

        aver_trends.loc[year]['data_7d_rol__mean'] = data_7d_rol['Consumption'][str(
            year)].mean()
        aver_trends.loc[year]['data_7d_rol__var'] = data_7d_rol['Consumption'][str(
            year)].std()
        aver_trends.loc[year]['data_7d_rol_eq__mean'] = data_correction['Consumption_eq_roll_7_day'][str(
            year)].mean()
        aver_trends.loc[year]['data_7d_rol_eq__var'] = data_correction['Consumption_eq_roll_7_day'][str(
            year)].std()

    fig, ax1 = plt.subplots(figsize=(14, 4))

    ax2 = ax1.twinx()
    ax1.plot(aver_trends['data_7d_rol__mean'],
             '--', label='data_7d_rol__mean')
    ax1.plot(aver_trends['data_7d_rol_eq__mean'],
             '--', label='data_7d_rol_eq__mean')
    ax2.plot(aver_trends['data_7d_rol__var'], label='data_7d_rol__var')
    ax2.plot(aver_trends['data_7d_rol_eq__var'],
             label='data_7d_rol_eq__var')

    ax1.set_xlabel('Years')
    ax1.set_ylabel('Mean data')
    ax2.set_ylabel('Variance data')
    fig.legend()
    plt.show()


def Create_time_predictors(df):

    df['datetime'] = df.index
    df['hour'] = df.datetime.dt.hour
    df['dayofweek'] = df.datetime.dt.dayofweek
    df['quarter'] = df.datetime.dt.quarter
    df['month'] = df.datetime.dt.month
    df['dayofyear'] = df.datetime.dt.dayofyear
    df['day'] = df.datetime.dt.day
    df['weekofyear'] = df.datetime.dt.isocalendar().week
    df = df.drop('datetime', axis=1)
    return df


def evaluate_time_series_models():

    data_correction = pd.read_parquet(
        'temp/enchanced_data_corrections/data_correction.parquet')

    # Добавление временных предикторов и очистка данных
    data_correction = Create_time_predictors(data_correction)
    columns_to_drop = ['Consumption', 'Wind', 'Solar', 'Wind+Solar',
                       'trend_residuals', 'trend_eq', 'Consumption_eq_roll_7_day', 'constant']
    data_correction = data_correction.drop(columns_to_drop, axis=1).dropna()

    # Разделение данных на обучающую и тестовую выборки
    X_train = data_correction[:'2016'].drop(['Consumption_eq'], axis=1)
    y_train = data_correction.loc[:'2016', 'Consumption_eq']
    X_test = data_correction['2017-01':'2017-12'].drop(
        ['Consumption_eq'], axis=1)
    y_test = data_correction.loc['2017-01':'2017-12', 'Consumption_eq']

    # Определение и оценка моделей
    models = [
        ('LR', LinearRegression()),
        ('NN', MLPRegressor(learning_rate_init=0.1)),
        ('KNN', KNeighborsRegressor()),
        ('RF', RandomForestRegressor(n_estimators=50)),
        ('SVR', SVR(gamma='auto', kernel='rbf'))
    ]
    results = []
    names = []

    # Тренировка и обучение
    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=10)
        cv_results = cross_val_score(
            model, X_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

    # Визуализация результатов
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    def regression_results(y_true, y_pred, model):
        mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
        mse = metrics.mean_squared_error(y_true, y_pred)
        median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)

    for name, model in models:
        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        regression_results(y_test, y_pred, model)

        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(X_test.index, y_pred, linewidth=2, label='Prediction')
        ax.plot(X_test.index, y_test, linewidth=2, label='Real')
        ax.set_title(f'Data Comparison {model}')
        ax.legend()
        ax.set_xlabel('Year')
        ax.set_ylabel('Consumption (GWh)')
        plt.show()

# 3 - Предсказание на 1 год


def predict_electricity_consumption(data=data):
    # Изменение данных
    data_consumption = data[['Consumption']].copy()
    data_consumption.loc[:, 'Yesterday'] = data_consumption.loc[:,
                                                                'Consumption'].shift()
    data_consumption.loc[:,
                         'Yesterday_Diff'] = data_consumption.loc[:, 'Yesterday'].diff()
    data_consumption = data_consumption.dropna()

    # Разделение данных на обучающую и тестовую выборки
    X_train = data_consumption.loc[:'2016'].drop(['Consumption'], axis=1)
    y_train = data_consumption.loc[:'2016', 'Consumption']
    X_test = data_consumption.loc['2017'].drop(['Consumption'], axis=1)

    # Обучение модели
    model = GradientBoostingRegressor(n_estimators=50)
    model.fit(X_train, y_train)

    # Предсказание на 2017 год
    y_pred = model.predict(X_test)

    # Загрузка фактических данных потребления электроэнергии на 2017 год из датафрейма
    y_actual = data.loc['2017', 'Consumption']

    # Вывод графика сравнения предсказанного и фактического потребления
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual.index, y_actual.values,
             label='Фактическое потребление', color='blue')
    plt.plot(X_test.index, y_pred,
             label='Предсказанное потребление', color='red')
    plt.title(
        'Сравнение фактического и предсказанного потребления электроэнергии на 2017 год')
    plt.xlabel('Дата')
    plt.ylabel('Потребление электроэнергии (GWh)')
    plt.legend()
    plt.show()

# 2 - Изменение размера тренировочных данных


def months_periods(data=data, model=LinearRegression()):

    # Изменение данных
    data_consumption = data[['Consumption']].copy()
    data_consumption.loc[:, 'Yesterday'] = data_consumption.loc[:,
                                                                'Consumption'].shift()
    data_consumption.loc[:,
                         'Yesterday_Diff'] = data_consumption.loc[:, 'Yesterday'].diff()
    data_consumption = data_consumption.dropna()

    # Инициализация списков для хранения результатов
    r2_scores = []
    mae_scores = []
    mse_scores = []
    rmse_scores = []

    # Цикл тренировки от 1 до 12 месяцев
    for period in range(1, 13):
        # Определение тренировочных и тестовых данных
        X_train = data_consumption[:f'2016-{period:02d}'].drop(
            ['Consumption'], axis=1)
        y_train = data_consumption.loc[:f'2016-{period:02d}', 'Consumption']
        X_test = data_consumption['2017-01':'2017-12'].drop(
            ['Consumption'], axis=1)
        y_test = data_consumption.loc['2017-01':'2017-12', 'Consumption']

        # Обучение модели
        model.fit(X_train, y_train)

        # Предсказание на тестовых данных
        y_pred = model.predict(X_test)

        # Рассчет метрик
        r2 = metrics.r2_score(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Сохранение метрик
        r2_scores.append(r2)
        mae_scores.append(mae)
        mse_scores.append(mse)
        rmse_scores.append(rmse)

    def plot_result(r2_scores, mae_scores, mse_scores, rmse_scores):
        metrics = {'R2': r2_scores, 'MAE': mae_scores,
                   'MSE': mse_scores, 'RMSE': rmse_scores}
        for metric, scores in metrics.items():
            plt.figure(figsize=(10, 6))
            plt.title(f'Зависимость метрик от размера тренировочных данных для модели {
                type(model).__name__}')
            plt.xlabel('Размер тренировочных данных (месяцев)')
            plt.ylabel(f'Метрика {metric}')
            plt.plot(range(1, 13), scores, label=metric)
            plt.legend()
            plt.show()

    plot_result(r2_scores, mae_scores, mse_scores, rmse_scores)


months_periods(data, RandomForestRegressor())
