import pandas as pd
from readCSV import readCsv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
from RandomForest import randomForest
from LogisticRegression import logisticRegression
from SGD import SGDModel
from DecisionTree import decisionTree
from GradientBoosting import gradientBoosting
from DataPreprocessing import dataPreprocessing
from CrossValidate import crossValidate
from Evaluatemodel import evaluate_model
from ModelComparison import modelComparison

app = FastAPI()

# 데이터 불러오기
df = readCsv()

# 데이터 전처리
preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)

# DTO
class Model(BaseModel):
    name : str
    parameters: dict[str, Any]
    

@app.get("/")
def home():
    return {"message": "Machine Learning API Server"}

@app.post("/analyze/all")
def analyzeAll():
    logistic_model = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    sgd_model = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_model = decisionTree(preprocessor, X_train, X_test, y_train, y_test)
    randomForest_model = randomForest(preprocessor, X_train, y_train, {"n_estimators": 200})
    gb_model = gradientBoosting(preprocessor, X_train, X_test, y_train, y_test)

    result = modelComparison(logistic_model, sgd_model, decisionTree_model, randomForest_model, gb_model, 
                             X_test, y_test, X_train, y_train)

    return result



@app.post("/analyze")
def analyze(data: Model):
    print("요청 모델:", data.name)
    print("요청 파라미터:", data.parameters)
    print("파라미터 타입:", type(data.parameters.get("n_estimators")))

    if data.name == "RandomForest":
        # 모델 생성, 학습
        rf_model = randomForest(preprocessor, X_train, y_train, data.parameters)

        # 교차검증 평가
        cv_result = crossValidate(data.name, rf_model, X_train, y_train)

        # 최종 테스트 평가
        test_result = evaluate_model(data.name, rf_model, X_test, y_test)

    elif data.name == "LogisticRegression":
        # 모델 생성, 학습
        lr_model = logisticRegression(preprocessor, X_train, y_train, data.parameters)
        
        # 교차검증 평가
        cv_result = crossValidate(data.name, lr_model, X_train, y_train)
        
        # 최종 테스트 평가
        test_result = evaluate_model(data.name, lr_model, X_test, y_test)

    elif data.name == "DecisionTree":
        # 모델 생성, 학습
        dt_model = decisionTree(preprocessor, X_train, y_train, data.parameters)
        
        # 교차검증 평가
        cv_result = crossValidate(data.name, dt_model, X_train, y_train)
        
        # 최종 테스트 평가
        test_result = evaluate_model(data.name, dt_model, X_test, y_test)

    elif data.name == "SGD":
        # 모델 생성, 학습
        sgd_model = SGDModel(preprocessor, X_train, y_train, data.parameters)
        
        # 교차검증 평가
        cv_result = crossValidate(data.name, sgd_model, X_train, y_train)
        
        # 최종 테스트 평가
        test_result = evaluate_model(data.name, sgd_model, X_test, y_test)

    elif data.name == "GradientBoosting":
        # 모델 생성, 학습
        gb_model = gradientBoosting(preprocessor, X_train, y_train, data.parameters)
        
        # 교차검증 평가
        cv_result = crossValidate(data.name, gb_model, X_train, y_train)
        
        # 최종 테스트 평가
        test_result = evaluate_model(data.name, gb_model, X_test, y_test)

    return {
        "cv_result": cv_result,
        "test_result": test_result
    }

