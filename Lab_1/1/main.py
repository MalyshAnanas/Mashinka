import pandas as pd
import matplotlib.pyplot as plt
import sys


def ReadDataFromFile(FileName):
    """
    Читает данные из CSV файла
    """
    try:
        data_frame = pd.read_csv(FileName)
        return data_frame
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit()


def DisplayStatisticalInfo(data_frame, column_x_name, column_y_name):
    """
    Выводит статистическую информацию для выбранных столбцов
    """
    print("Статистическая информация по данным:")
    print(f"Столбец X ({column_x_name}):")
    print(f"  Количество: {data_frame[column_x_name].count()}")
    print(f"  Минимальное значение: {data_frame[column_x_name].min()}")
    print(f"  Максимальное значение: {data_frame[column_x_name].max()}")
    print(f"  Среднее значение: {data_frame[column_x_name].mean()}")
    print(f"\nСтолбец Y ({column_y_name}):")
    print(f"  Количество: {data_frame[column_y_name].count()}")
    print(f"  Минимальное значение: {data_frame[column_y_name].min()}")
    print(f"  Максимальное значение: {data_frame[column_y_name].max()}")
    print(f"  Среднее значение: {data_frame[column_y_name].mean()}")


def PlotOriginalPoints(data_frame, column_x_name, column_y_name, ax):
    """
    Отрисовывает исходные точки на графике.
    """
    ax.scatter(data_frame[column_x_name], data_frame[column_y_name], label='Исходные точки', color='blue')
    ax.set_xlabel(column_x_name)
    ax.set_ylabel(column_y_name)
    ax.set_title('Исходные данные и линия регрессии')
    ax.legend()


def CalculateRegressionParameters(data_frame, column_x_name, column_y_name):
    """
    Реализует метод наименьших квадратов для вычисления параметров регрессионной прямой
    """
    x = data_frame[column_x_name]
    y = data_frame[column_y_name]

    # Вычисление параметров методом наименьших квадратов
    n = len(x)
    SumOfX = x.sum()
    SumOfY = y.sum()
    SumOfXY = (x * y).sum()
    SumOfXSquare = (x ** 2).sum()

    SlopeOfRegression = (n * SumOfXY - SumOfX * SumOfY) / (n * SumOfXSquare - SumOfX ** 2)
    InterceptOfRegression = (SumOfY - SlopeOfRegression * SumOfX) / n

    return SlopeOfRegression, InterceptOfRegression


def PlotRegressionLine(data_frame, column_x_name, slope, intercept, ax):
    """
    Отрисовывает регрессионную прямую на графике.
    """
    x = data_frame[column_x_name]
    # Вычисление предсказанных значений Y
    PredictedY = slope * x + intercept
    ax.plot(x, PredictedY, color='red', label=f'Линия регрессии (y = {slope:.2f}x + {intercept:.2f})')
    ax.legend()


def PlotErrorSquares(data_frame, column_x_name, column_y_name, slope, intercept, ax):
    """
    Отрисовывает и заштриховывает квадраты ошибок.
    """
    x_values = data_frame[column_x_name].values
    y_values = data_frame[column_y_name].values
    PredictedY_values = slope * x_values + intercept

    ax.scatter(x_values, y_values, color='blue', label='Исходные точки')
    ax.plot(x_values, PredictedY_values, color='red', label='Линия регрессии')
    ax.set_xlabel(column_x_name)
    ax.set_ylabel(column_y_name)
    ax.set_title('Квадраты ошибок')
    ax.legend()

    # Отрисовка квадратов ошибок
    for i in range(len(x_values)):
        actual_y = y_values[i]
        predicted_y = PredictedY_values[i]
        error = actual_y - predicted_y

        # Координаты квадрата ошибки
        # Если ошибка отрицательная, квадрат строится влево
        SquareX = [x_values[i], x_values[i], x_values[i] + error, x_values[i] + error, x_values[i]]
        SquareY = [predicted_y, actual_y, actual_y, predicted_y, predicted_y]

        # Отрисовка и заштриховывание квадрата
        ax.fill(SquareX, SquareY, color='orange', alpha=0.3, hatch='///')
        # Соединяем исходную точку с точкой на линии регрессии
        ax.plot([x_values[i], x_values[i]], [actual_y, predicted_y], color='gray', linestyle='--')


def PerformLinearRegression(FileName, ColumnForX, ColumnForY):
    data_frame = ReadDataFromFile(FileName)

    if ColumnForX not in data_frame.columns or ColumnForY not in data_frame.columns:
        print(f"Ошибка: Указанные столбцы '{ColumnForX}' или '{ColumnForY}' не найдены в файле.")
        print(f"Доступные столбцы: {', '.join(data_frame.columns)}")
        sys.exit()

    DisplayStatisticalInfo(data_frame, ColumnForX, ColumnForY)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    PlotOriginalPoints(data_frame, ColumnForX, ColumnForY, axs[0])

    SlopeOfLine, InterceptOfLine = CalculateRegressionParameters(data_frame, ColumnForX, ColumnForY)
    print(f"\nПараметры регрессионной прямой:")
    print(f"  Угловой коэффициент: {SlopeOfLine:.4f}")
    print(f"  Свободный член: {InterceptOfLine:.4f}")

    PlotOriginalPoints(data_frame, ColumnForX, ColumnForY, axs[1])
    PlotRegressionLine(data_frame, ColumnForX, SlopeOfLine, InterceptOfLine, axs[1])
    axs[1].set_title('Исходные данные и линия регрессии (для квадратов ошибок)')

    PlotErrorSquares(data_frame, ColumnForX, ColumnForY, SlopeOfLine, InterceptOfLine, axs[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FileNameToUse = 'student_scores.csv'
    ColumnForXToUse = 'Hours'
    ColumnForYToUse = 'Scores'

    PerformLinearRegression(FileNameToUse, ColumnForXToUse, ColumnForYToUse)
