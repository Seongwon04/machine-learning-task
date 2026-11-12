import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from Evaluatemodel import evaluate_model

def modelComparison(logistic_model, sgd_model, tree_model, rf_model, gb_model, X_test, y_test):
    # 모델 목록 구성
    models = {
    "Logistic Regression": logistic_model,
    "SGD Classifier": sgd_model,
    "Decision Tree": tree_model,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
    }

    # 모델별 평가 실행
    results = []
    for name, model in models.items():
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)
        result_df = pd.DataFrame(results)
    
    print(result_df.round(4))

    # 모델별 성능 시각화
    result_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1"]
    ].plot(
        kind="bar",
        figsize=(12, 6)
    )
    plt.title("Machine Learning Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
