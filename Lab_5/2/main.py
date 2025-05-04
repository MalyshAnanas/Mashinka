import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import time
import xgboost as xgb


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
    Разбивает на обучающую и тестовую выборку
    """

    X = data_frame.drop(columns=[target_column])
    y = data_frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def EvaluateModelMetrics(model, X_test, y_test, model_name):
    """
    Выводит метрики качества модели
    """
    print(f"\nМетрики для модели: {model_name}")

    PredictedY = model.predict(X_test)

    Accuracy = accuracy_score(y_test, PredictedY)
    Precision = precision_score(y_test, PredictedY)
    Recall = recall_score(y_test, PredictedY)
    F1 = f1_score(y_test, PredictedY)

    ProbaY = model.predict_proba(X_test)[:, 1]
    RocAuc = roc_auc_score(y_test, ProbaY)

    print(f"ROC AUC: {RocAuc:.4f}")
    print(f"Точность (Accuracy): {Accuracy:.4f}")
    print(f"Точность (Precision): {Precision:.4f}")
    print(f"Полнота (Recall): {Recall:.4f}")
    print(f"F1-score: {F1:.4f}")

    return {'Accuracy': Accuracy, 'Precision': Precision, 'Recall': Recall, 'F1': F1, 'ROC_AUC': RocAuc}


def TrainRandomForestModel(X_train, y_train, n_estimators=100, max_depth=None, max_features='sqrt', random_state=0):
    """
    Обучает модель случайного леса
    """
    ModelRF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                     max_features=max_features, random_state=random_state, n_jobs=-1)
    ModelRF.fit(X_train, y_train)
    return ModelRF


def TrainXGBoostModel(X_train, y_train, params, random_state=0):
    """
    Обучает модель XGBoost
    """
    ModelXGB = xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, **params)
    ModelXGB.fit(X_train, y_train)

    return ModelXGB


def InvestigateRFParameter(X_train, X_test, y_train, y_test, param_name, param_values, metric_name):
    """
    Исследует значение метрики в зависимости от значения заданного параметра
    """
    print(f"\nИсследование метрики '{metric_name}' в зависимости от параметра '{param_name}' случайного леса")
    metric_values = []
    training_times = []

    for param_value in param_values:
        start_time = time.time()
        if param_name == 'max_depth':
            ModelRF = TrainRandomForestModel(X_train, y_train, max_depth=param_value)
        elif param_name == 'max_features':
            if isinstance(param_value, str) or param_value is None or param_value > X_train.shape[1]:
                actual_max_features = param_value
            else:
                actual_max_features = int(param_value)
            ModelRF = TrainRandomForestModel(X_train, y_train, max_features=actual_max_features)
        elif param_name == 'n_estimators':
            ModelRF = TrainRandomForestModel(X_train, y_train, n_estimators=param_value)

        end_time = time.time()
        training_time = end_time - start_time
        training_times.append(training_time)

        if metric_name == 'Accuracy':
            metric_value = accuracy_score(y_test, ModelRF.predict(X_test))
        elif metric_name == 'Precision':
            metric_value = precision_score(y_test, ModelRF.predict(X_test))
        elif metric_name == 'Recall':
            metric_value = recall_score(y_test, ModelRF.predict(X_test))
        elif metric_name == 'F1':
            metric_value = f1_score(y_test, ModelRF.predict(X_test))
        elif metric_name == 'ROC_AUC':
            ProbaY = ModelRF.predict_proba(X_test)[:, 1]
            metric_value = roc_auc_score(y_test, ProbaY)

        metric_values.append(metric_value)
        print(
            f"Параметр '{param_name}' = {param_value}: {metric_name} = {metric_value:.4f}, время обучения = {training_time:.4f} сек")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(param_values, metric_values, marker='o', color='blue')
    ax1.set_xlabel(f'Значение параметра "{param_name}"')
    ax1.set_ylabel(f'Значение метрики ({metric_name})', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    plt.title(f'Зависимость {metric_name} от параметра "{param_name}" Random Forest')
    plt.grid(True)

    if param_name == 'n_estimators':
        ax2 = ax1.twinx()
        ax2.plot(param_values, training_times, marker='x', color='red')
        ax2.set_ylabel('Время обучения (сек)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        # plt.title(f'Зависимость {metric_name} и времени обучения от {param_name} Random Forest')

    plt.show()

    return metric_values, training_times


def EvaluateXGBoost(X_train, X_test, y_train, y_test, params, model_name="XGBoost"):
    """
    Обучает модель XGBoost
    """
    print(f"\nОценка модели: {model_name}")

    start_time = time.time()
    ModelXGB = TrainXGBoostModel(X_train, y_train, params)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Время обучения: {training_time:.4f} сек")

    metrics = EvaluateModelMetrics(ModelXGB, X_test, y_test, model_name)

    return ModelXGB, metrics, training_time


def CompareModelsConclusion(metrics_rf, metrics_xgb, training_time_xgb):
    """
    Сравнивает метрики и время обучения моделей
    """
    MetricsDF = pd.DataFrame({
        "Random Forest": metrics_rf,
        "XGBoost": metrics_xgb
    }).T
    print(MetricsDF)


def PerformLab():
    DiabetesFileName = 'diabetes.csv'

    diabetes_df = LoadDiabetesDataset(DiabetesFileName)

    TargetColumn = diabetes_df.columns.tolist()[-1]
    print(f"Целевая колонка: '{TargetColumn}'")
    print("Первые 5 строк загруженного датасета:")
    print(diabetes_df.head())

    if diabetes_df.drop(columns=[TargetColumn]).shape[1] == 0:
        print(
            "Ошибка: После определения целевой колонки не осталось признаков для обучения. Проверьте название целевой колонки.")
        sys.exit()

    X_train, X_test, y_train, y_test = SplitDataForClassification(diabetes_df, TargetColumn)

    print("\nКлассификация методом случайного леса и исследование гиперпараметров")

    DepthsToInvestigateRF = range(1, 11)
    InvestigateRFParameter(X_train, X_test, y_train, y_test, 'max_depth', DepthsToInvestigateRF, metric_name='F1')

    FeaturesToInvestigateRF = range(1, X_train.shape[1] + 1)
    InvestigateRFParameter(X_train, X_test, y_train, y_test, 'max_features', FeaturesToInvestigateRF,
                           metric_name='F1')

    NestimatorsToInvestigateRF = range(10, 201, 20)
    InvestigateRFParameter(
        X_train, X_test, y_train, y_test, 'n_estimators', NestimatorsToInvestigateRF, metric_name='F1'
    )

    OptimalRFMaxDepth = 6
    OptimalRFMaxFeatures = 2
    OptimalRFNestimators = 70

    print(f"\nОбучение финальной модели Random Forest с параметрами")
    print(f"  max_depth: {OptimalRFMaxDepth}")
    print(f"  max_features: {OptimalRFMaxFeatures}")
    print(f"  n_estimators: {OptimalRFNestimators}")

    ModelRF_Optimal = TrainRandomForestModel(X_train, y_train,
                                             n_estimators=OptimalRFNestimators,
                                             max_depth=OptimalRFMaxDepth,
                                             max_features=OptimalRFMaxFeatures)
    metrics_rf_optimal = EvaluateModelMetrics(ModelRF_Optimal, X_test, y_test, "Random Forest (оптимальный)")

    print("\nКлассификация с использованием XGBoost и исследование")

    XGBoostParams = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100
    }
    ModelXGB, metrics_xgb, training_time_xgb = EvaluateXGBoost(X_train, X_test, y_train, y_test, params=XGBoostParams)

    CompareModelsConclusion(metrics_rf_optimal, metrics_xgb, training_time_xgb)


if __name__ == "__main__":
    PerformLab()
