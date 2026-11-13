import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from RandomForest import randomForest

def gridSearchCV(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    # 랜덤 포레스트 모델 생성
    rf_model = randomForest(preprocessor, X_train, X_test, y_train, y_test)

    # 파라미터 후보 설정
    params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 7, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
    }

    # GridSearchCV 생성
    grid = GridSearchCV(
    estimator=rf_model,
    param_grid=params,
    cv=5,
    scoring="recall",
    n_jobs=4,
    verbose=1
    )

    # 최적 모델 학습
    grid.fit(X_train, y_train)

    print("최적 파라미터")
    print(grid.best_params_)
    print("최고 교차 검증 Recall")
    print(grid.best_score_)

    # 최적 모델 성능 확인
    best_model = grid.best_estimator_
    best_pred = best_model.predict(X_test)

    result = {
            "Accuracy": accuracy_score(y_test, best_pred),
            "Precision": precision_score(y_test, best_pred),
            "Recall": recall_score(y_test, best_pred),
            "F1": f1_score(y_test, best_pred),
            "ROC-AUC:": roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
        }

    result_df = pd.DataFrame([result])
    print("최적 모델 평가")
    print(result_df.round(4))

    


