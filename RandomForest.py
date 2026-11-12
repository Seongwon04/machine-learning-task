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

def randomForest(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    # 모델 생성
    rf_model = Pipeline([("preprocessor", preprocessor),("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=42,
        n_jobs=-1
        ))
    ])

    # 모델 학습
    rf_model.fit(X_train,y_train)
    print("랜덤 포레스트 학습 완료")

    # 예측
    rf_pred = rf_model.predict(X_test)

    # 모델 평가
    print("Accuracy : ", accuracy_score(y_test, rf_pred))
    print("Precision : ", precision_score(y_test, rf_pred))
    print("Recall : ", recall_score(y_test, rf_pred))
    print("F1-score : ", f1_score(y_test, rf_pred))
    print("Log Loss : ", log_loss(y_test, rf_pred))
    result = classification_report(y_test, rf_pred)
    return result
