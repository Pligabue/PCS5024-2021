from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def get_stats(y, predicted_y):
    
    res = {
        "TP": ((predicted_y == ">50K") & (y == ">50K")).sum(),
        "TN": ((predicted_y == "<=50K") & (y == "<=50K")).sum(),
        "FP": ((predicted_y == ">50K") & (y == "<=50K")).sum(),
        "FN": ((predicted_y == "<=50K") & (y == ">50K")).sum()
    }
    
    display(pd.Series(res))
    
    stats = {
        "Precision": res["TP"]/(res["TP"] + res["FP"]),
        "Recall": res["TP"]/(res["TP"] + res["FN"]),
        "F1-score": 2*res["TP"]/(2*res["TP"] + res["FP"] + res["FN"]),
        "Accuracy": accuracy_score(y, predicted_y)
    }
    
    display(pd.Series(stats))
    