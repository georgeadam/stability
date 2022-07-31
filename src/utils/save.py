import pandas as pd


def save_predictions(original_preds, new_preds):
    predictions = pd.DataFrame({"original": original_preds, "new": new_preds})
    predictions.to_csv("predictions.csv")