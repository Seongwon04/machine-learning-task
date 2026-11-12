import pandas as pd
from readCSV import readCsv
from CsvInfo import printInfo
from AnalysisAndVisualization import AAV
from DataPreprocessing import dataPreprocessing
from LogisticRegression import logisticRegression
from SGD import SGDModel
from DecisionTree import decisionTree
from RandomForest import randomForest

def main():
    df = readCsv()

    #printInfo(df)
    #AAV(df)
    preprocessor, X_train, X_test, y_train, y_test = dataPreprocessing(df)
    """
    logistic_pred = logisticRegression(preprocessor, X_train, X_test, y_train, y_test)    
    ssg_pred = SGDModel(preprocessor, X_train, X_test, y_train, y_test)    
    """
    #decisionTree(preprocessor, X_train, X_test, y_train, y_test)
    randomForest(preprocessor, X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()