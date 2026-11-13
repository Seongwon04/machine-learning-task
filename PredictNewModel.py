import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def predictNewModel(model, new_customer:pd.DataFrame):
    # 고객 이탈 예측
    prediction = model.predict(new_customer)
    probability = model.predict_proba(new_customer)

    churn_probability = probability[0][1]

    print("\n\n\n고객 이탈 예측 결과")
    if prediction[0] == 1:
        print("이탈 예상 고객")
    else:
        print("유지 예상 고객")

    print(f"이탈 확률: {churn_probability * 100:.2f}%")
