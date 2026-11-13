import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from typing import Any

def randomForest(preprocessor:ColumnTransformer, X_train:pd.DataFrame, y_train:pd.DataFrame, parameters:dict[str, Any]):
    # 모델 생성
    rf_model = Pipeline([("preprocessor", preprocessor),("model", RandomForestClassifier(
        n_estimators=parameters["n_estimators"],
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        min_samples_split=5
        ))
    ])

    # 모델 학습
    rf_model.fit(X_train,y_train)
    print("랜덤 포레스트 학습 완료")

    return rf_model
