import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # Опорные векторы
from sklearn.neighbors import KNeighborsClassifier  # Ближайшие соседи
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, precision_recall_curve, roc_curve, \
    roc_auc_score
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


def TrainLogisticRegressionModel(X_train, y_train):
    """
    Обучает модель
    """
    ModelLR = LogisticRegression(random_state=0, solver='liblinear')
    ModelLR.fit(X_train, y_train)
    return ModelLR


def TrainSVMModel(X_train, y_train):
    """
    Обучает модель опорных векторов (SVM)
    """
    ModelSVM = SVC(random_state=0, probability=True)
    ModelSVM.fit(X_train, y_train)
    return ModelSVM


def TrainKNNModel(X_train, y_train, n_neighbors=5):
    """
    Обучает модель ближайших соседей (kN)
    """
    ModelKNN = KNeighborsClassifier(n_neighbors=n_neighbors)
    ModelKNN.fit(X_train, y_train)
    return ModelKNN


def EvaluateAndVisualizeModel(model, X_test, y_test, model_name):
    """
    Показывает метрики
    """
    print(f"\nОценка модели: {model_name}")

    PredictedY = model.predict(X_test)

    Accuracy = accuracy_score(y_test, PredictedY)
    Precision = precision_score(y_test, PredictedY)
    Recall = recall_score(y_test, PredictedY)
    F1 = f1_score(y_test, PredictedY)

    print(f"Точность (Accuracy): {Accuracy:.4f}")
    print(f"Точность (Precision): {Precision:.4f}")
    print(f"Полнота (Recall): {Recall:.4f}")
    print(f"F1-score: {F1:.4f}")

    cm = confusion_matrix(y_test, PredictedY)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="cool")
    plt.title(f'Матрица ошибок для {model_name}')
    plt.show()

    Precision_curve, Recall_curve, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(Recall_curve, Precision_curve, marker='.')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title(f'PR-кривая для {model_name}')
    plt.grid(True)
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label=f'ROC AUC = {roc_auc:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-кривая для {model_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return Accuracy, Precision, Recall, F1, roc_auc


def CompareModels(metrics_dict):
    """
    Сравнивает модели
    """
    print("\nСравнение моделей")
    MetricsDF = pd.DataFrame(metrics_dict).T
    print(MetricsDF)

    print("\nВывод:")
    BestModelName = MetricsDF['F1'].idxmax()
    print(f"Наилучшая модель по F1-score: {BestModelName}")
    BestModelName_Accuracy = MetricsDF['Accuracy'].idxmax()
    print(f"Наилучшая модель по Accuracy: {BestModelName_Accuracy}")
    BestModelName_ROC_AUC = MetricsDF['ROC_AUC'].idxmax()
    print(f"Наилучшая модель по ROC AUC: {BestModelName_ROC_AUC}")


def PerformClassificationMetricsLab():
    TitanicFileName = 'Titanic.csv'
    titanic_df = LoadTitanicDataset(TitanicFileName)

    ProcessedTitanicDF, PercentageLost = PreprocessTitanicData(titanic_df)
    print(f"\nПроцент потерянных данных после предобработки: {PercentageLost:.2f}%")
    print(f"Размер датасета после предобработки: {ProcessedTitanicDF.shape}")
    print("Первые 5 строк предобработанного датасета:")
    print(ProcessedTitanicDF.head())

    if 'Survived' in ProcessedTitanicDF.columns and ProcessedTitanicDF.shape[1] > 1:
        X_train, X_test, y_train, y_test = SplitDataForClassification(ProcessedTitanicDF, 'Survived')

        MetricsForComparison = {}

        ModelLR = TrainLogisticRegressionModel(X_train, y_train)
        metrics_lr = EvaluateAndVisualizeModel(ModelLR, X_test, y_test, "Логистическая регрессия")
        MetricsForComparison["Логистическая регрессия"] = {
            'Accuracy': metrics_lr[0], 'Precision': metrics_lr[1], 'Recall': metrics_lr[2], 'F1': metrics_lr[3],
            'ROC_AUC': metrics_lr[4]
        }

        ModelSVM = TrainSVMModel(X_train, y_train)
        metrics_svm = EvaluateAndVisualizeModel(ModelSVM, X_test, y_test, "SVM")
        MetricsForComparison["SVM"] = {
            'Accuracy': metrics_svm[0], 'Precision': metrics_svm[1], 'Recall': metrics_svm[2], 'F1': metrics_svm[3],
            'ROC_AUC': metrics_svm[4]
        }

        ModelKNN = TrainKNNModel(X_train, y_train, n_neighbors=5)
        metrics_knn = EvaluateAndVisualizeModel(ModelKNN, X_test, y_test, "kNN (n=5)")
        MetricsForComparison["kNN (n=5)"] = {
            'Accuracy': metrics_knn[0], 'Precision': metrics_knn[1], 'Recall': metrics_knn[2], 'F1': metrics_knn[3],
            'ROC_AUC': metrics_knn[4]
        }

        CompareModels(MetricsForComparison)


if __name__ == "__main__":
    PerformClassificationMetricsLab()
