import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sys


def LoadTitanicDataset(FileName):
    """
    Загружает набор данных Titanic из CSV файла
    """
    try:
        data_frame = pd.read_csv(FileName)
        return data_frame
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit()


def PreprocessTitanicData(data_frame):
    """
    Предобработка данных датасета Titanic:
    """
    InitialRowCount = data_frame.shape[0]

    # Удаление строк с пропусками
    ProcessedDataFrame = data_frame.dropna().copy()

    # Удаление столбцов с нечисловыми значениями, кроме Sex и Embarked
    ColumnsToDrop = ['Name', 'Ticket', 'Cabin']
    ProcessedDataFrame = ProcessedDataFrame.drop(columns=ColumnsToDrop)

    # Изменить данные в числовой вид в столбцах Sex и Embarked
    # female=0, male=1
    ProcessedDataFrame['Sex'] = ProcessedDataFrame['Sex'].map({'female': 0, 'male': 1})
    # Embarked: C=0, Q=1, S=2
    EmbarkedMapping = {'C': 0, 'Q': 1, 'S': 2}
    ProcessedDataFrame['Embarked'] = ProcessedDataFrame['Embarked'].map(EmbarkedMapping)

    ProcessedDataFrame = ProcessedDataFrame.drop(columns=['PassengerId'])

    FinalRowCount = ProcessedDataFrame.shape[0]
    PercentageLost = ((InitialRowCount - FinalRowCount) / InitialRowCount) * 100

    return ProcessedDataFrame, PercentageLost


def SplitDataForClassification(data_frame, target_column):
    """
    Разбивает данные на учебную и тестовую выборку
    """
    X = data_frame.drop(columns=[target_column])
    y = data_frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def TrainAndEvaluateLogisticRegression(X_train, X_test, y_train, y_test, dataset_name=""):
    """
    Обучает модель
    """
    ModelLR = LogisticRegression(random_state=0, solver='liblinear')

    ModelLR.fit(X_train, y_train)

    PredictedY = ModelLR.predict(X_test)

    Accuracy = accuracy_score(y_test, PredictedY)
    print(f"Точность модели{f' на датасете {dataset_name}' if dataset_name else ''}: {Accuracy:.4f}")

    return ModelLR


def AssessEmbarkedImpact(data_frame, target_column):
    """
    Влияние 'Embarked' на модель
    """
    print("\n")

    X_with_embarked = data_frame.drop(columns=[target_column])
    y = data_frame[target_column]
    X_train_with, X_test_with, y_train_with, y_test_with = train_test_split(X_with_embarked, y, test_size=0.2,
                                                                            random_state=42, stratify=y)
    TrainAndEvaluateLogisticRegression(X_train_with, X_test_with, y_train_with, y_test_with, "с Embarked")

    X_without_embarked = data_frame.drop(columns=[target_column, 'Embarked'])
    if X_without_embarked.shape[1] == 0:
        print("Ошибка: После удаления Embarked не осталось признаков для обучения модели.")
        return
    y = data_frame[target_column]
    X_train_without, X_test_without, y_train_without, y_test_without = train_test_split(X_without_embarked, y,
                                                                                        test_size=0.2, random_state=42,
                                                                                        stratify=y)
    TrainAndEvaluateLogisticRegression(X_train_without, X_test_without, y_train_without, y_test_without, "без Embarked")


def LoadIrisDataset():
    """
    Загружает датасет Iris.
    """
    iris_dataset = datasets.load_iris()
    return iris_dataset


def PrepareDataForMulticlass(iris_dataset):
    """
    Обработка данных из датасета Iris
    """
    X = iris_dataset.data
    y = iris_dataset.target
    return X, y


def PlotMulticlassDecisionBoundary(X, y, feature_names, target_names):
    """
    Отрисовывает данные
    """
    FeatureIndex1 = feature_names.index('petal length (cm)')
    FeatureIndex2 = feature_names.index('petal width (cm)')
    X_plot = X[:, [FeatureIndex1, FeatureIndex2]]

    ModelLR_Plotting = LogisticRegression(random_state=0, solver='liblinear', multi_class='auto')
    ModelLR_Plotting.fit(X_plot, y)

    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = ModelLR_Plotting.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, cmap="cool", alpha=0.8)

    scatter = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap="cool", edgecolor='k', s=50)
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.title('Многоклассовая логистическая регрессия на датасете Iris')

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label=target_names[i],
                                 markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10)
                      for i in np.unique(y)]
    plt.legend(handles=legend_handles, title="Сорта")

    plt.show()


def PerformLab():
    # Titanic
    TitanicFileName = 'titanic.csv'
    titanic_df = LoadTitanicDataset(TitanicFileName)

    if titanic_df is not None:
        ProcessedTitanicDF, PercentageLost = PreprocessTitanicData(titanic_df)
        print(f"Процент потерянных данных после предобработки: {PercentageLost:.2f}%")
        print(f"Размер датасета после предобработки: {ProcessedTitanicDF.shape}")
        print("Первые 5 строк предобработанного датасета:")
        print(ProcessedTitanicDF.head())

        X_train_titanic, X_test_titanic, y_train_titanic, y_test_titanic = SplitDataForClassification(
            ProcessedTitanicDF, 'Survived')
        TrainAndEvaluateLogisticRegression(X_train_titanic, X_test_titanic, y_train_titanic,
                                           y_test_titanic, "Titanic")

        if 'Embarked' in ProcessedTitanicDF.columns:
            AssessEmbarkedImpact(ProcessedTitanicDF, 'Survived')

    # Часть 2: Iris
    iris_dataset = LoadIrisDataset()
    X_iris, y_iris = PrepareDataForMulticlass(iris_dataset)

    PlotMulticlassDecisionBoundary(X_iris, y_iris, iris_dataset.feature_names, iris_dataset.target_names)


if __name__ == "__main__":
    PerformLab()
