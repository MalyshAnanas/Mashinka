import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    precision_recall_curve
import sys


def LoadDiabetesDataset(FileName):
    """
    Загружает датасет о диабете
    """
    try:
        data_frame = pd.read_csv(FileName)
        return data_frame
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        sys.exit()


def SplitDataForClassification(data_frame, target_column):
    """
    Разбивает обучающую и тестовую выборку
    """
    X = data_frame.drop(columns=[target_column])
    y = data_frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def TrainLogisticRegressionModel(X_train, y_train):
    """
    Обучает модель логистической регрессии
    """
    ModelLR = LogisticRegression(random_state=0, solver='liblinear')
    ModelLR.fit(X_train, y_train)
    return ModelLR


def TrainDecisionTreeModel(X_train, y_train, max_depth=None):
    """
    Обучает модель решающего дерева
    """
    ModelTree = DecisionTreeClassifier(random_state=0, max_depth=max_depth)
    ModelTree.fit(X_train, y_train)
    return ModelTree


def EvaluateModelMetrics(model, X_test, y_test, model_name):
    """
    Метрики моделей
    """
    print(f"\nМетрики для модели: {model_name}")

    PredictedY = model.predict(X_test)

    Accuracy = accuracy_score(y_test, PredictedY)
    Precision = precision_score(y_test, PredictedY)
    Recall = recall_score(y_test, PredictedY)
    F1 = f1_score(y_test, PredictedY)

    if hasattr(model, "predict_proba"):
        try:
            ProbaY = model.predict_proba(X_test)[:, 1]
            RocAuc = roc_auc_score(y_test, ProbaY)
            print(f"ROC AUC: {RocAuc:.4f}")
        except Exception as e:
            RocAuc = np.nan
            print(f"Не удалось вычислить ROC AUC: {e}")
    else:
        RocAuc = np.nan
        print("Модель не поддерживает predict_proba для расчета ROC AUC.")

    print(f"Точность (Accuracy): {Accuracy:.4f}")
    print(f"Точность (Precision): {Precision:.4f}")
    print(f"Полнота (Recall): {Recall:.4f}")
    print(f"F1-score: {F1:.4f}")

    return {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1, 'ROC_AUC': RocAuc}


def InvestigateTreeDepth(X_train, X_test, y_train, y_test, max_depths, metric_name='Accuracy'):
    """
    Исследует значение метрики в зависимости от максимальной глубины решающего дерева.
    Строит график зависимости. Возвращает список значений метрики для каждой глубины и оптимальную глубину.
    """
    print(f"\nИсследование метрики '{metric_name}' в зависимости от глубины дерева")
    print(
        "Accuracy - наиболее простая и понятная метрика, она позволяет понять общее качество модели"
        " без акцентирования на детали")
    metric_values = []
    best_metric_value = -1
    optimal_depth = None

    for depth in max_depths:
        ModelTree = TrainDecisionTreeModel(X_train, y_train, max_depth=depth)
        PredictedY = ModelTree.predict(X_test)

        if metric_name == 'Accuracy':
            metric_value = accuracy_score(y_test, PredictedY)
        elif metric_name == 'Precision':
            metric_value = precision_score(y_test, PredictedY)
        elif metric_name == 'Recall':
            metric_value = recall_score(y_test, PredictedY)
        elif metric_name == 'F1':
            metric_value = f1_score(y_test, PredictedY)
        elif metric_name == 'ROC_AUC':
            if hasattr(ModelTree, "predict_proba"):
                try:
                    ProbaY = ModelTree.predict_proba(X_test)[:, 1]
                    metric_value = roc_auc_score(y_test, ProbaY)
                except:
                    metric_value = np.nan
            else:
                metric_value = np.nan
        else:
            print(f"Ошибка: Неподдерживаемая метрика '{metric_name}'.")
            return None, None

        metric_values.append(metric_value)
        print(f"Глубина {depth}: {metric_name} = {metric_value:.4f}")

        if not np.isnan(metric_value) and metric_value > best_metric_value:
            best_metric_value = metric_value
            optimal_depth = depth

    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, metric_values, marker='o')
    plt.xlabel('Максимальная глубина дерева')
    plt.ylabel(f'Значение метрики ({metric_name})')
    plt.title(f'Зависимость {metric_name} от глубины решающего дерева')
    plt.grid(True)
    plt.xticks(max_depths)
    plt.show()

    print(f"\nОптимальная глубина дерева по метрике '{metric_name}': {optimal_depth}")

    return metric_values, optimal_depth


def ExportDecisionTreeGraphviz(model, feature_names, class_names, dot_file_name="tree"):
    """
    Экспортирует решающее дерево в формат .dot для Graphviz
    """
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(c) for c in class_names],
                               filled=True, rounded=True,
                               special_characters=True)

    try:
        with open(f"{dot_file_name}.dot", "w") as dot_file:
            dot_file.write(dot_data)
    except Exception as e:
        print(f"\nОшибка при экспорте дерева в файл: {e}")


def PlotFeatureImportances(model, feature_names):
    """
    Отрисовывает важность признаков модели решающего дерева
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Важность признаков")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()


def PlotROCCurve(model, X_test, y_test, model_name):
    """
    Отрисовывает ROC
    """
    ProbaY = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, ProbaY)
    roc_auc = roc_auc_score(y_test, ProbaY)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='violet', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC для {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def PlotPRCurve(model, X_test, y_test, model_name):
    """
    Отрисовывает PR
    """
    ProbaY = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, ProbaY)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Полнота (Recall)')
    plt.ylabel('Точность (Precision)')
    plt.title(f'PR-кривая для {model_name}')
    plt.grid(True)
    plt.show()


def PerformLab():
    DiabetesFileName = 'diabetes.csv'

    diabetes_df = LoadDiabetesDataset(DiabetesFileName)

    TargetColumn = diabetes_df.columns[-1]
    print(f"Целевая колонка: '{TargetColumn}'")
    print("Первые 5 строк загруженного датасета:")
    print(diabetes_df.head())

    X_train, X_test, y_train, y_test = SplitDataForClassification(diabetes_df, TargetColumn)

    print("\nКлассификация Логистической регрессией")
    ModelLR = TrainLogisticRegressionModel(X_train, y_train)
    EvaluateModelMetrics(ModelLR, X_test, y_test, "Логистическая регрессия")

    print("\nКлассификация Решающим деревом (стандартные настройки)")
    ModelTree_Standard = TrainDecisionTreeModel(X_train, y_train)
    EvaluateModelMetrics(ModelTree_Standard, X_test, y_test, "Решающее дерево (стандартные настройки)")

    DepthsToInvestigate = range(1, 11)
    metric_values_depth, OptimalDepth = InvestigateTreeDepth(
        X_train, X_test, y_train, y_test, DepthsToInvestigate
    )

    print(f"\nВизуализация для Решающего дерева с оптимальной глубиной ({OptimalDepth})")
    ModelTree_Optimal = TrainDecisionTreeModel(X_train, y_train, max_depth=OptimalDepth)

    ExportDecisionTreeGraphviz(ModelTree_Optimal, X_train.columns, y_train.unique(),
                               dot_file_name=f"decision_tree_depth_{OptimalDepth}")

    PlotFeatureImportances(ModelTree_Optimal, X_train.columns)

    PlotPRCurve(ModelTree_Optimal, X_test, y_test, f"Решающее дерево (глубина={OptimalDepth})")
    PlotROCCurve(ModelTree_Optimal, X_test, y_test, f"Решающее дерево (глубина={OptimalDepth})")


if __name__ == "__main__":
    PerformLab()
