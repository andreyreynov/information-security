import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

my_column_names = ['Lead Guitar', 'Rhythm Guitar', 'Bass', 'Drums', 'Vocals']
my_data = np.random.randint(low=10, high=50, size=(10, 5))
df = pd.DataFrame(data=my_data, columns=my_column_names)


def plot_histogram(row_index):
    plt.bar(my_column_names, df.iloc[row_index])
    plt.xlabel('Инструменты')
    plt.ylabel('Кол-во выступлений')
    plt.title(f'Выбрана строка: {row_index}')
    plt.show()


def statistics(dataframe):
    statistics = dataframe.describe()
    return statistics


def date_time():
    date_index = pd.date_range(start='2024-2-18', periods=10, freq='D')
    df = pd.DataFrame(data=my_data, columns=my_column_names, index=date_index)
    print(df)


def trasported_df():
    df_transposed = df.T

    print("Исходный датафрейм:")
    print(df)
    print("\nТранспонированный датафрейм:")
    print(df_transposed)


def sum_two_columns(dataframe, col1, col2):
    dataframe['sum'] = dataframe[col1] + dataframe[col2]
    return dataframe


# plot_histogram(3)
# print(statistics(df))
# trasported_df()
# date_time()
# print(sum_two_columns(df, 'Lead Guitar', 'Rhythm Guitar'))
