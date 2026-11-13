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

def main():
    df = readCsv()

    #printInfo(df)
    #AAV(df)
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)

    logistic_model = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    sgd_model = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    decisionTree_model = decisionTree(preprocessor, X_train, X_test, y_train, y_test)
    randomForest_model = randomForest(preprocessor, X_train, X_test, y_train, y_test)
    gb_model = gradientBoosting(preprocessor, X_train, X_test, y_train, y_test)

    #modelComparison(logistic_model, sgd_model, decisionTree_model, randomForest_model, gb_model, X_test, y_test)

    gridSearchCV(preprocessor, X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()