import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def SGDModel(preprocessor:ColumnTransformer, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.DataFrame, y_test:pd.DataFrame):
    # 모델 생성
    sgd_model = Pipeline([("preprocessor", preprocessor),("model", SGDClassifier(loss="log_loss", max_iter=2000, random_state=42))])

    # 모델 학습
    sgd_model.fit(X_train, y_train)
    print("SGD 학습 완료")

    return sgd_model