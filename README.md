# AdaBoosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Генерация синтетических данных для примера
np.random.seed(0)
X = np.random.rand(100, 2)  # 100 абитуриентов с двумя признаками
y = np.random.randint(2, size=100)  # Метки класса (0 - не принят, 1 - принят)

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация базовой модели (решающее дерево с ограничением глубины)
base_model = DecisionTreeClassifier(max_depth=1)

adaboost_model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1.0)

adaboost_model.fit(X_train, y_train)

y_pred = adaboost_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")
