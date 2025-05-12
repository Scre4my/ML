import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Загрузка данных
df = pd.read_csv('Titanic.csv')
print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

# 📊 EDA (Exploratory Data Analysis)
print("\n=== 📊 EDA ===")

# 1. Анализ пропущенных значений
print("\nПропущенные значения:")
print(df.isnull().sum().sort_values(ascending=False))

# 2. Распределение выживших
plt.figure(figsize=(8, 5))
sns.countplot(x='Survived', data=df)
plt.title('Распределение выживших (0 - Погиб, 1 - Выжил)')
plt.show()
print("\nВывод: Несбалансированное распределение - большинство пассажиров не выжило.")

# 3. Выживаемость по полу
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Процент выживших по полу')
plt.show()
print("\nВывод: Женщины имели значительно более высокие шансы на выживание.")

# 4. Выживаемость по классу каюты
plt.figure(figsize=(8, 5))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Процент выживших по классу каюты')
plt.show()
print("\nВывод: Пассажиры 1-го класса имели наивысшие шансы на выживание.")

# 5. Распределение возраста
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Распределение возраста пассажиров')
plt.show()
print("\nВывод: Большинство пассажиров были в возрасте 20-40 лет.")

# 6. Выживаемость по возрасту
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, multiple='stack')
plt.title('Распределение возраста по выжившим и погибшим')
plt.show()
print("\nВывод: Дети до 10 лет имели повышенные шансы на выживание.")

# 7. Корреляция признаков с целевой переменной
df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1})
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_num']
corr_with_target = df[numeric_features + ['Survived']].corr()['Survived'].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=corr_with_target.index, y=corr_with_target.values)
plt.title('Корреляция признаков с выживаемостью')
plt.xticks(rotation=45)
plt.show()
print("\nВывод: Пол и стоимость билета имеют наибольшую корреляцию с выживаемостью.")

# 👨‍💻 Feature Engineering
print("\n=== 👨‍💻 Feature Engineering ===")

# 1. Создание новых признаков
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Age*Class'] = df['Age'] * df['Pclass']
df['FarePerPerson'] = df['Fare'] / df['FamilySize']

# 2. Обработка категориальных признаков
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
df['Title'] = df['Title'].replace('Mme', 'Mrs')

# 3. Заполнение пропущенных значений
df['Age'] = df.groupby(['Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# 4. Проверка корреляции новых признаков
new_features = ['FamilySize', 'IsAlone', 'Age*Class', 'FarePerPerson']
df_new = df.copy()
df_new['Sex'] = df_new['Sex'].map({'male': 0, 'female': 1})
df_new['Title'] = df_new['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4})

print("\nКорреляция новых признаков с выживаемостью:")
print(df_new[new_features + ['Survived']].corr()['Survived'].sort_values(ascending=False))

# 👩‍🎓 Подготовка данных для моделирования
print("\n=== 👩‍🎓 Подготовка данных ===")

# Выбор признаков
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
X = df[features]
y = df['Survived']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Препроцессинг
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'IsAlone', 'Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 👩‍🎓 Сравнение моделей
print("\n=== 👩‍🎓 Сравнение моделей ===")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    # Создание пайплайна
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Кросс-валидация
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

    # Обучение на полном тренировочном наборе
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    results[name] = {
        'CV Mean Accuracy': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Accuracy': test_accuracy
    }

# Вывод результатов
results_df = pd.DataFrame(results).T.sort_values(by='CV Mean Accuracy', ascending=False)
print("\nРезультаты кросс-валидации и тестирования:")
print(results_df)

# Выбор лучшей модели
best_model_name = results_df.index[0]
print(f"\nЛучшая модель: {best_model_name}")

# Обучение лучшей модели на всех данных
best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])
final_pipeline.fit(X, y)

# Важность признаков для лучшей модели
if hasattr(best_model, 'feature_importances_'):
    # Для моделей с feature_importances_
    ohe_columns = list(final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
        categorical_features))
    all_features = numeric_features + ohe_columns
    importances = pd.Series(best_model.feature_importances_, index=all_features)

    plt.figure(figsize=(12, 8))
    importances.sort_values().plot(kind='barh')
    plt.title('Важность признаков')
    plt.show()
elif hasattr(best_model, 'coef_'):
    # Для логистической регрессии
    ohe_columns = list(final_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(
        categorical_features))
    all_features = numeric_features + ohe_columns
    coefs = pd.Series(best_model.coef_[0], index=all_features)

    plt.figure(figsize=(12, 8))
    coefs.sort_values().plot(kind='barh')
    plt.title('Коэффициенты модели')
    plt.show()

# Отчет по лучшей модели
print("\nОтчет по лучшей модели:")
y_pred = final_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Матрица ошибок
print("\nМатрица ошибок:")
print(confusion_matrix(y_test, y_pred))

# Итоговый вывод
print("\n=== Итоговый вывод ===")
print("1. Лучшая модель:", best_model_name)
print(f"   - Средняя точность на кросс-валидации: {results_df.loc[best_model_name, 'CV Mean Accuracy']:.3f}")
print(f"   - Точность на тестовом наборе: {results_df.loc[best_model_name, 'Test Accuracy']:.3f}")

print("\n2. Ключевые факторы выживания:")
print("   - Пол (женщины имели значительно больше шансов)")
print("   - Класс каюты (1-й класс имел преимущество)")
print("   - Возраст (дети выживали чаще)")
print("   - Размер семьи (одиночки имели меньшие шансы)")

print("\n3. Рекомендации для улучшения:")
print("   - Более тщательная обработка пропусков в возрасте")
print("   - Эксперименты с другими архитектурами нейросетей")
print("   - Подбор гиперпараметров для моделей")
print("   - Добавление внешних данных (например, о расположении кают)")