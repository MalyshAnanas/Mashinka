import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification


def LoadIrisDataset():
    """
    Загружает набор данных Iris
    """
    iris_dataset = datasets.load_iris()
    return iris_dataset


def ExploreIrisDataset(iris_dataset):
    """
    Отрисовывает зависимости
    """
    iris_df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
    iris_df['target'] = iris_dataset.target
    iris_df['species'] = iris_df['target'].apply(lambda x: iris_dataset.target_names[x])

    print("Первые 5 строк датасета Iris:")
    print(iris_df.head())
    print("\nНазвания сортов:", iris_dataset.target_names)
    print("\nЗначения целевой переменной:", np.unique(iris_dataset.target))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='species', data=iris_df, ax=axs[0],
                    palette='cool')
    axs[0].set_title('Зависимость sepal length от sepal width')
    axs[0].set_xlabel('sepal length')
    axs[0].set_ylabel('sepal width')

    sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='species', data=iris_df, ax=axs[1],
                    palette='cool')
    axs[1].set_title('Зависимость petal length от petal width')
    axs[1].set_xlabel('petal length')
    axs[1].set_ylabel('petal width')

    plt.tight_layout()
    plt.show()

    print("\nОтрисовка pairplot для датасета Iris:")
    sns.pairplot(iris_df, hue='species', palette='cool')
    plt.suptitle('Pairplot для датасета Iris', y=1.02)
    plt.show()

    return iris_df


def PrepareBinaryDatasets(iris_df):
    """
    Готовит датасеты для бинарной классификации
    """
    dataset_setosa_versicolor = iris_df[iris_df['target'].isin([0, 1])].copy()
    dataset_setosa_versicolor.reset_index(drop=True, inplace=True)
    print("\nДатасет: setosa и versicolor")
    print(dataset_setosa_versicolor['species'].value_counts())

    dataset_versicolor_virginica = iris_df[iris_df['target'].isin([1, 2])].copy()
    dataset_versicolor_virginica.reset_index(drop=True, inplace=True)
    print("\nДатасет: versicolor и virginica")
    print(dataset_versicolor_virginica['species'].value_counts())

    return dataset_setosa_versicolor, dataset_versicolor_virginica


def SplitDataset(data_frame, target_column):
    """
    Разбивает на обучающую и тестовую выборку
    """
    X = data_frame.drop([target_column, 'species'], axis=1)
    y = data_frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y)

    print(f"\nРазмер обучающей выборки X: {X_train.shape}")
    print(f"Размер тестовой выборки X: {X_test.shape}")
    print(f"Размер обучающей выборки y: {y_train.shape}")
    print(f"Размер тестовой выборки y: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def TrainAndEvaluateLogisticRegression(X_train, X_test, y_train, y_test, dataset_name):
    """
    Обучает и предсказывает
    """
    print(f"\nМодель логистической регрессии для датасета: {dataset_name}")

    ModelLR = LogisticRegression(random_state=0)

    ModelLR.fit(X_train, y_train)
    print("Модель успешно обучена.")

    PredictedY = ModelLR.predict(X_test)

    Accuracy = accuracy_score(y_test, PredictedY)
    print(f"Точность модели на тестовой выборке: {Accuracy:.4f}")

    return ModelLR


def GenerateSyntheticDataset():
    """
    Генерирует датасет
    """
    print("\nГенерация синтетического датасета")
    X_synth, y_synth = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                                           random_state=1, n_clusters_per_class=1)
    print(f"Размер сгенерированного датасета X: {X_synth.shape}")
    print(f"Размер сгенерированного датасета y: {y_synth.shape}")

    return X_synth, y_synth


def PlotSyntheticDataset(X_synth, y_synth):
    """
    Отрисовывает сгенерированный датасет
    """
    print("\nОтрисовка сгенерированного датасета:")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_synth[:, 0], X_synth[:, 1], c=y_synth, cmap='cool', marker='o', edgecolor='k')
    plt.title('Сгенерированный синтетический датасет для бинарной классификации')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.colorbar(label='Класс')
    plt.grid(True)
    plt.show()


def PerformBinaryClassificationLab():
    print("Часть 1: Бинарная классификация на датасете Iris")
    iris_dataset = LoadIrisDataset()
    iris_df = ExploreIrisDataset(iris_dataset)

    dataset_setosa_versicolor, dataset_versicolor_virginica = PrepareBinaryDatasets(iris_df)

    X_train_sv, X_test_sv, y_train_sv, y_test_sv = SplitDataset(dataset_setosa_versicolor, 'target')
    TrainAndEvaluateLogisticRegression(X_train_sv, X_test_sv, y_train_sv, y_test_sv, "setosa vs versicolor")

    X_train_vv, X_test_vv, y_train_vv, y_test_vv = SplitDataset(dataset_versicolor_virginica, 'target')
    TrainAndEvaluateLogisticRegression(X_train_vv, X_test_vv, y_train_vv, y_test_vv, "versicolor vs virginica")

    print("\nЧасть 2: Бинарная классификация на сгенерированном датасете")
    X_synth, y_synth = GenerateSyntheticDataset()
    PlotSyntheticDataset(X_synth, y_synth)

    synth_df = pd.DataFrame(X_synth, columns=['feature_1', 'feature_2'])
    synth_df['target'] = y_synth
    synth_df['species'] = synth_df['target'].astype(str)

    X_train_synth, X_test_synth, y_train_synth, y_test_synth = SplitDataset(synth_df, 'target')
    TrainAndEvaluateLogisticRegression(X_train_synth, X_test_synth, y_train_synth, y_test_synth,
                                       "сгенерированный датасет")


if __name__ == "__main__":
    PerformBinaryClassificationLab()
