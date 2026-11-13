import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from typing import Any

def logisticRegression(preprocessor:ColumnTransformer, X_train:pd.DataFrame, y_train:pd.DataFrame, parameters:dict[str, Any]):
    # 모델 생성
    logistic_model = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, random_state=42, C=parameters["C"]))])
    # 모델 학습
    logistic_model.fit(X_train, y_train)
    print("로지스틱 회귀 학습 완료")

    return logistic_model