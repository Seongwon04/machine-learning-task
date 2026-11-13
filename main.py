import pandas as pd
from readCSV import readCsv
from CsvInfo import printInfo
from AnalysisAndVisualization import AAV
from DataPreprocessing import dataPreprocessing
from LogisticRegression import logisticRegression
from SGD import SGDModel
from DecisionTree import decisionTree
from RandomForest import randomForest
from GradientBoosting import gradientBoosting
from ModelComparison import modelComparison
from GridSearchCV import gridSearchCV
from PredictNewModel import predictNewModel

def main():
    # 데이터 불러오기
    df = readCsv()

    #printInfo(df)
    #AAV(df)

    # 데이터 전처리
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)

    # 모델 생성, 학습
    logistic_model = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    sgd_model = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_model = decisionTree(preprocessor, X_train, X_test, y_train, y_test)
    randomForest_model = randomForest(preprocessor, X_train, X_test, y_train, y_test)
    gb_model = gradientBoosting(preprocessor, X_train, X_test, y_train, y_test)

    # 모델 비교
    #modelComparison(logistic_model, sgd_model, decisionTree_model, randomForest_model, gb_model, X_test, y_test)

    # 최적 모델 찾기
    #gridSearchCV(preprocessor, X_train, X_test, y_train, y_test)

    # 최적 모델로 새로운 고객 예측
    new_customer = pd.DataFrame([
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0
    }])

    predictNewModel(randomForest_model, new_customer)



    
if __name__ == "__main__":
    main()