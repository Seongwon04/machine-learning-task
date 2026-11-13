import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from typing import Any

def decisionTree(preprocessor:ColumnTransformer, X_train:pd.DataFrame, y_train:pd.DataFrame, parameters:dict[str, Any]):
    tree_model = Pipeline([("preprocessor", preprocessor), 
                           ("model", DecisionTreeClassifier(random_state=42, max_depth=parameters["max_depth"]))])
    tree_model.fit(X_train, y_train)

    return tree_model