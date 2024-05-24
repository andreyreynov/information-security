import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# Шаг 1. Загрузка данных
penguins = sns.load_dataset('penguins')

# Шаг 2. Подготовка данных
penguins.dropna(inplace=True)  # Удаляем строки с отсутствующими значениями
X = penguins.drop('species', axis=1)  # Все признаки, кроме 'species'
y = penguins['species']  # Целевая переменная

# Преобразование категориальных признаков в числовые
label_encoder = LabelEncoder()
X['island'] = label_encoder.fit_transform(X['island'])
X['sex'] = label_encoder.fit_transform(X['sex'])

# Создание общего окна для всех матриц несоответствия
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Шаг 3. Цикл для классификации с разным количеством признаков
for num_features, ax in zip(range(2, 8), axes.flatten()):
    if num_features <= 6:
        X_train, X_test, y_train, y_test = train_test_split(
            X.iloc[:, :num_features], y, test_size=0.2, random_state=42)

        # Обучение модели
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Предсказание на тестовом наборе данных
        y_pred = model.predict(X_test)

        # Оценка точности
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Точность при {num_features} признаках: {accuracy}')

        # Шаг 4: Матрица несоответствия
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f'Матрица несоответствия при {num_features} признаках:')
        print(conf_matrix)

        # Отображение матриц с помощью ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_matrix, display_labels=model.classes_)
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title(f'{num_features} признаков')
        ax.set_xlabel('Предсказание')
        ax.set_ylabel('Оригинал')
        print('-----------------------------')

plt.tight_layout()
plt.show()
