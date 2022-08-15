import pandas as pd


def save_predictions(original_train_preds, original_test_preds, new_train_preds, new_test_preds):
    predictions = pd.DataFrame(
        {"original_train": original_train_preds, "original_test": original_test_preds, "new_train": new_train_preds,
         "new_test": new_test_preds})
    predictions.to_csv("predictions.csv")
