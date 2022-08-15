def compute_accuracy(preds, labels):
    return (preds == labels).astype(float).mean()


def compute_overall_churn(original_preds, new_preds):
    return (original_preds != new_preds).astype(float).mean()


def compute_relevant_churn(original_preds, new_preds, labels):
    return ((original_preds == labels) & (new_preds != original_preds)).astype(float).mean()
