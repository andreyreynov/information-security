import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score

penguins = sns.load_dataset('penguins')

# Удаляем строки с отсутствующими значениями
penguins.dropna(inplace=True)
# Все признаки, кроме 'species'
X = penguins.drop('species', axis=1)
# Целевая переменная
y = penguins['species']

# Преобразование категориальных признаков в числовые
label_encoder = LabelEncoder()
X['island'] = label_encoder.fit_transform(X['island'])
X['sex'] = label_encoder.fit_transform(X['sex'])

for num_features in range(2, 7):
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
    print('-----------------------------')
