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


def log_final_metrics_frankenstein(test_labels, meta_test_preds, frankenstein_test_preds, new_test_preds, original_test_preds):
    meta_overall_churn = compute_overall_churn(original_test_preds, meta_test_preds)
    meta_relevant_churn = compute_relevant_churn(original_test_preds, meta_test_preds, test_labels)
    meta_accuracy = compute_accuracy(meta_test_preds, test_labels)

    frankenstein_overall_churn = compute_overall_churn(original_test_preds, frankenstein_test_preds)
    frankenstein_relevant_churn = compute_relevant_churn(original_test_preds, frankenstein_test_preds, test_labels)
    frankenstein_accuracy = compute_accuracy(frankenstein_test_preds, test_labels)

    new_overall_churn = compute_overall_churn(original_test_preds, new_test_preds)
    new_relevant_churn = compute_relevant_churn(original_test_preds, new_test_preds, test_labels)
    new_accuracy = compute_accuracy(new_test_preds, test_labels)

    original_accuracy = compute_accuracy(original_test_preds, test_labels)

    wandb_logger = WandbLogger(project="stability")
    wandb_logger.log_metrics({"test/meta_overall_churn": meta_overall_churn,
                              "test/meta_relevant_churn": meta_relevant_churn,
                              "test/meta_accuracy": meta_accuracy,
                              "test/frankenstein_overall_churn": frankenstein_overall_churn,
                              "test/frankenstein_relevant_churn": frankenstein_relevant_churn,
                              "test/frankenstein_accuracy": frankenstein_accuracy,
                              "test/new_overall_churn": new_overall_churn,
                              "test/new_relevant_churn": new_relevant_churn,
                              "test/new_accuracy": new_accuracy,
                              "test/original_accuracy": original_accuracy
                              })