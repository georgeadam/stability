from pytorch_lightning.loggers import WandbLogger

from src.utils.metrics import compute_overall_churn, compute_relevant_churn, compute_accuracy


def log_final_metrics(dataset, new_test_preds, new_train_preds, original_test_preds, original_train_preds):
    # Compute churn
    train_labels, test_labels = dataset.train_labels, dataset.test_labels
    overall_train_churn = compute_overall_churn(original_train_preds, new_train_preds)
    overall_test_churn = compute_overall_churn(original_test_preds, new_test_preds)
    relevant_train_churn = compute_relevant_churn(original_train_preds, new_train_preds, train_labels)
    relevant_test_churn = compute_relevant_churn(original_test_preds, new_test_preds, test_labels)
    # Compute accuracy
    original_train_accuracy = compute_accuracy(original_train_preds, train_labels)
    original_test_accuracy = compute_accuracy(original_test_preds, test_labels)
    new_train_accuracy = compute_accuracy(new_train_preds, train_labels)
    new_test_accuracy = compute_accuracy(new_test_preds, test_labels)
    wandb_logger = WandbLogger(project="stability")
    wandb_logger.log_metrics({"train/overall_churn": overall_train_churn,
                              "train/relevant_churn": relevant_train_churn,
                              "train/original_accuracy": original_train_accuracy,
                              "train/new_accuracy": new_train_accuracy,
                              "test/overall_churn": overall_test_churn,
                              "test/relevant_churn": relevant_test_churn,
                              "test/original_accuracy": original_test_accuracy,
                              "test/new_accuracy": new_test_accuracy
                              })
