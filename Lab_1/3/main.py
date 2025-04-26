import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_absolute_error, r2_score
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


def CalculateAndDisplayMetrics(ActualY, PredictedY, model_name):
    """
    Вычисляет и выводит метрики MAE, R2, MAPE.
    """
    mae = mean_absolute_error(ActualY, PredictedY)
    r2 = r2_score(ActualY, PredictedY)

    # Расчет MAPE вручную для надежности, если mean_absolute_percentage_error недоступна
    # Избегаем деления на ноль, если фактическое значение равно 0
    mape = np.mean(np.abs((ActualY - PredictedY) / ActualY)) * 100
    # Заменим Inf на NaN, если есть деление на ноль ActualY
    mape = np.where(np.isinf(mape), np.nan, mape)
    # Удалим NaN значения для расчета среднего MAPE
    mape = np.nanmean(mape)

    print(f"\nМетрики качества модели ({model_name}):")
    print(f"  Средняя абсолютная ошибка (MAE): {mae:.4f}")
    print(f"  Коэффициент детерминации (R2): {r2:.4f}")
    # Проверяем, является ли mape числом перед форматированием
    if np.isnan(mape):
        print(
            f"  Средняя абсолютная процентная ошибка (MAPE): Невозможно вычислить (есть нулевые фактические значения Y)")
    else:
        print(f"  Средняя абсолютная процентная ошибка (MAPE): {mape:.4f}%")


def PerformDiabetesLinearRegressionWithMetrics():
    diabetes_dataset = LoadDiabetesDataset()

    FeatureData, TargetData, FeatureName = ExploreDatasetAndChooseFeature(diabetes_dataset)

    CustomSlope, CustomIntercept = CalculateRegressionParametersCustom(FeatureData, TargetData)

    if CustomSlope is not None and CustomIntercept is not None:
        print(f"\nКоэффициенты собственного алгоритма:")
        print(f"  Угловой коэффициент (a): {CustomSlope:.4f}")
        print(f"  Свободный член (b): {CustomIntercept:.4f}")
        # Предсказания для собственного алгоритма
        CustomPredictedY = CustomSlope * FeatureData.flatten() + CustomIntercept
    else:
        CustomPredictedY = None

    SKLearnModel = PerformLinearRegressionSKLearn(FeatureData, TargetData)

    print(f"\nКоэффициенты Scikit-Learn:")
    print(
        f"  Угловой коэффициент (coef_): {SKLearnModel.coef_[0]:.4f}")
    print(f"  Свободный член (intercept_): {SKLearnModel.intercept_:.4f}")

    # Предсказания для Scikit-Learn
    SKLearnPredictedY = SKLearnModel.predict(FeatureData)

    PlotRegressionResults(FeatureData, TargetData, FeatureName, CustomSlope, CustomIntercept, SKLearnModel)

    DisplayPredictionTable(FeatureData, TargetData, SKLearnModel, FeatureName)

    CalculateAndDisplayMetrics(TargetData, CustomPredictedY, "Свой алгоритм")

    CalculateAndDisplayMetrics(TargetData, SKLearnPredictedY, "Scikit-Learn")


# Пример использования:
if __name__ == "__main__":
    PerformDiabetesLinearRegressionWithMetrics()
