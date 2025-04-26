import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression


def LoadDiabetesDataset():
    """
    Загружает diabetes из Scikit-Learn
    """
    diabetes_dataset = datasets.load_diabetes()
    return diabetes_dataset


def ExploreDatasetAndChooseFeature(diabetes_dataset):
    """
    Возвращает выбранный столбец признака и целевую переменную
    """

    IndexForFeature = diabetes_dataset.feature_names.index('bmi')
    FeatureData = diabetes_dataset.data[:, np.newaxis, IndexForFeature]
    TargetData = diabetes_dataset.target

    print("Выбран признак:", diabetes_dataset.feature_names[IndexForFeature])
    print("Форма данных признака:", FeatureData.shape)
    print("Форма целевой переменной:", TargetData.shape)

    return FeatureData, TargetData, diabetes_dataset.feature_names[IndexForFeature]


def CalculateRegressionParametersCustom(FeatureData, TargetData):
    """
    Метод наименьших квадратов для вычисления параметров
    """
    x = FeatureData.flatten()
    y = TargetData

    n = len(x)
    SumOfX = x.sum()
    SumOfY = y.sum()
    SumOfXY = (x * y).sum()
    SumOfXSquare = (x ** 2).sum()

    Denominator = n * SumOfXSquare - SumOfX ** 2

    SlopeOfRegression = (n * SumOfXY - SumOfX * SumOfY) / Denominator
    InterceptOfRegression = (SumOfY - SlopeOfRegression * SumOfX) / n

    return SlopeOfRegression, InterceptOfRegression


def PerformLinearRegressionSKLearn(FeatureData, TargetData):
    """
    Линейная регрессия Scikit-Learn
    """

    ModelSKLearn = LinearRegression()

    ModelSKLearn.fit(FeatureData, TargetData)

    return ModelSKLearn


def PlotRegressionResults(FeatureData, TargetData, FeatureName, CustomSlope, CustomIntercept, SKLearnModel):
    """
    Отрисовывает исходные данные и регрессионные прямые алгоритмов
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(FeatureData, TargetData, color='blue', label='Исходные данные')

    # Собственный алгоритм
    if CustomSlope is not None and CustomIntercept is not None:
        CustomPredictedY = CustomSlope * FeatureData.flatten() + CustomIntercept
        plt.plot(FeatureData, CustomPredictedY, color='red', label='Регрессия (свой алгоритм)')

    # Scikit-Learn
    SKLearnPredictedY = SKLearnModel.predict(FeatureData)
    plt.plot(FeatureData, SKLearnPredictedY, color='violet', label='Регрессия (Scikit-Learn)')

    plt.xlabel(FeatureName)
    plt.ylabel('Целевая переменная (прогрессия диабета)')
    plt.title('Сравнение линейной регрессии: Свой алгоритм vs Scikit-Learn')
    plt.legend()
    plt.grid(True)
    plt.show()


def DisplayPredictionTable(FeatureData, TargetData, SKLearnModel, FeatureName):
    """
    Таблица с исходными данными
    """
    SKLearnPredictedY = SKLearnModel.predict(FeatureData)
    ErrorsOfPrediction = TargetData - SKLearnPredictedY

    PredictionResults = pd.DataFrame({
        FeatureName: FeatureData.flatten(),
        'Фактическое Y': TargetData,
        'Предсказанное Y (SKLearn)': SKLearnPredictedY,
        'Ошибка': ErrorsOfPrediction
    })

    print("\nТаблица результатов предсказаний (Scikit-Learn):")

    print(PredictionResults.head())
    print("...")
    print(PredictionResults.tail())


def PerformDiabetesLinearRegression():
    diabetes_dataset = LoadDiabetesDataset()

    FeatureData, TargetData, FeatureName = ExploreDatasetAndChooseFeature(diabetes_dataset)

    CustomSlope, CustomIntercept = CalculateRegressionParametersCustom(FeatureData, TargetData)

    if CustomSlope is not None and CustomIntercept is not None:
        print(f"\nКоэффициенты собственного алгоритма:")
        print(f"  Угловой коэффициент (a): {CustomSlope:.4f}")
        print(f"  Свободный член (b): {CustomIntercept:.4f}")

    SKLearnModel = PerformLinearRegressionSKLearn(FeatureData, TargetData)

    print(f"\nКоэффициенты Scikit-Learn:")
    print(
        f"  Угловой коэффициент (coef_): {SKLearnModel.coef_[0]:.4f}")
    print(f"  Свободный член (intercept_): {SKLearnModel.intercept_:.4f}")

    PlotRegressionResults(FeatureData, TargetData, FeatureName, CustomSlope, CustomIntercept, SKLearnModel)

    DisplayPredictionTable(FeatureData, TargetData, SKLearnModel, FeatureName)


# Пример использования:
if __name__ == "__main__":
    PerformDiabetesLinearRegression()
