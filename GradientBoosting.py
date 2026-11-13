import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from typing import Any

def gradientBoosting(preprocessor:ColumnTransformer, X_train:pd.DataFrame, y_train:pd.DataFrame, parameters:dict[str, Any]):
    # 모델 생성
    gb_model = Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=parameters["learning_rate"],
            max_depth=3,
            random_state=42
        ))
    ])

    # 모델 학습
    gb_model.fit(X_train, y_train)
    print("그래이디언트 부스팅 학습 완료")

    return gb_model




