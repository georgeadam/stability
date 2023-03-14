import logging
import os
from collections import OrderedDict

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.neighbors import NearestNeighbors

from settings import ROOT_DIR
from src.combiners import combiners
from src.data import datasets
from src.lightning_modules import lightning_modules
from src.models import models
from src.utils.hydra import get_wandb_run
from src.utils.inference import extract_predictions
from src.utils.metrics import compute_accuracy, compute_relevant_churn

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


def remove_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "original_" in k:
            continue

        name = k.replace("model.", "")
        new_state_dict[name] = v

    return new_state_dict


def extract_probs_and_preds(dataloader, model):
    module = lightning_modules.create("prediction", model=model)
    trainer = Trainer(gpus=1, enable_checkpointing=False, enable_progress_bar=False)

    logits = trainer.predict(module, dataloaders=dataloader)
    logits = torch.cat(logits)
    preds = torch.argmax(logits, dim=1)
    probs = torch.nn.Softmax(dim=1)(logits)
    sample_indices = torch.arange(len(probs))

    return probs, probs[sample_indices, preds].detach().cpu().numpy(), preds.detach().cpu().numpy()


def get_checkpoints(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))

    new_index = 0

    for i in range(1, len(checkpoints)):
        if checkpoints[i].startswith("epoch=0"):
            new_index = i
            break

    base_checkpoints = checkpoints[:new_index]
    new_checkpoints = checkpoints[new_index:]

    return base_checkpoints, new_checkpoints


def get_all_checkpoint_predictions(config, dataset, dataloader, checkpoint_dir, checkpoints):
    all_probabilities = []
    all_confidences = []
    all_predictions = []

    for checkpoint_file in checkpoints:
        model = models.create(config.model.name, num_classes=dataset.num_classes, num_channels=dataset.num_channels,
                              height=dataset.height, **config.model.params)
        checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_file))
        state_dict = remove_prefix(checkpoint["state_dict"])

        model.load_state_dict(state_dict)
        model.eval()

        probabilities, confidences, predictions = extract_probs_and_preds(dataloader, model)
        all_probabilities.append(probabilities)
        all_confidences.append(confidences)
        all_predictions.append(predictions)

    all_probabilities = np.stack(all_probabilities, axis=1)
    all_confidences = np.stack(all_confidences, axis=1)
    all_predictions = np.stack(all_predictions, axis=1)

    return all_probabilities, all_confidences, all_predictions


def compute_meta_choice_stats(dataset, base_predictions, new_predictions, meta_choices):
    combined_predictions = np.stack([base_predictions, new_predictions], axis=1)
    sample_indices = np.arange(len(combined_predictions))
    meta_predictions_combined = combined_predictions[sample_indices, meta_choices]
    meta_acc_combined = compute_accuracy(meta_predictions_combined, dataset.test_labels)
    meta_churn_combined = compute_relevant_churn(base_predictions, meta_predictions_combined, dataset.test_labels)

    return meta_acc_combined, meta_churn_combined


def get_combination_stats(dataset, base_predictions, new_predictions, base_conf_scores, new_conf_scores,
                          base_avg_conf_scores, new_avg_conf_scores):
    meta_choices = ((new_avg_conf_scores > base_avg_conf_scores) & (new_conf_scores > base_conf_scores)).astype(int)

    return compute_meta_choice_stats(dataset, base_predictions, new_predictions, meta_choices)


def get_single_stats(dataset, base_predictions, new_predictions, base_scores, new_scores):
    meta_choices = (new_scores > base_scores).astype(int)

    return compute_meta_choice_stats(dataset, base_predictions, new_predictions, meta_choices)


def compute_conf_score(confidences):
    return confidences[:, -1]


def compute_avg_conf_score(probabilities, predictions):
    sample_indices = np.arange(len(probabilities))

    return np.mean(probabilities[sample_indices, :, predictions[:, -1]], axis=1)


@torch.no_grad()
def extract_embeddings(dataloader, model):
    embeddings = []

    for x, _, _, _ in dataloader:
        embedding = model.forward_embedding(x)
        embeddings.append(embedding)

    return torch.cat(embeddings).detach().cpu().numpy()


def compute_estimated_avg_conf_score(config, dataset, model, checkpoint_dir, checkpoints):
    validation_embeddings = extract_embeddings(dataset.val_dataloader(), model)
    test_embeddings = extract_embeddings(dataset.test_dataloader(), model)
    nearest_neighbors = NearestNeighbors(n_neighbors=10)
    nearest_neighbors.fit(validation_embeddings)

    neighbor_indices = nearest_neighbors.kneighbors(test_embeddings)[1]
    validation_probabilities, _, validation_predictions = get_all_checkpoint_predictions(config, dataset,
                                                                                         dataset.val_dataloader(),
                                                                                         checkpoint_dir, checkpoints)

    validation_avg_conf_scores = compute_avg_conf_score(validation_probabilities, validation_predictions)
    test_avg_conf_scores = validation_avg_conf_scores[neighbor_indices]

    return np.mean(test_avg_conf_scores, axis=1)


@hydra.main(config_path=config_path, config_name="flip_count")
def main(args: DictConfig):
    # So we want to measure a couple of things
    # 1. How correlated the kNN score is with the actual AvgConf
    # 2. How the churn reduction ability of the two scores varies.
    # For both, we'll need to extract the predicted probabilities for every checkpoint.
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    experiment_dir = os.path.join(ROOT_DIR, args.experiment_dir)
    checkpoint_dir = os.path.join(experiment_dir, "stability/{}/checkpoints".format(args.run_id))

    experiment_config = OmegaConf.load(os.path.join(experiment_dir, "config.yaml"))

    seed_everything(experiment_config.misc.seed, workers=True)
    dataset = datasets.create(experiment_config.data.name, **experiment_config.data.params)

    base_checkpoints, new_checkpoints = get_checkpoints(checkpoint_dir)

    model_base = models.create(experiment_config.model.name, num_classes=dataset.num_classes,
                               num_channels=dataset.num_channels, height=dataset.height,
                               **experiment_config.model.params)
    model_new = models.create(experiment_config.model.name, num_classes=dataset.num_classes,
                              num_channels=dataset.num_channels, height=dataset.height,
                              **experiment_config.model.params)
    checkpoint_base = torch.load(os.path.join(checkpoint_dir, base_checkpoints[-1]))
    checkpoint_new = torch.load(os.path.join(checkpoint_dir, new_checkpoints[-1]))
    state_dict_base = remove_prefix(checkpoint_base["state_dict"])
    state_dict_new = remove_prefix(checkpoint_new["state_dict"])

    model_base.load_state_dict(state_dict_base)
    model_base.eval()

    model_new.load_state_dict(state_dict_new)
    model_new.eval()

    base_final_predictions = extract_predictions(dataset.test_dataloader(), model_base)
    new_final_predictions = extract_predictions(dataset.test_dataloader(), model_new)

    base_probabilities, base_confidences, base_predictions = get_all_checkpoint_predictions(experiment_config, dataset,
                                                                                            dataset.test_dataloader(),
                                                                                            checkpoint_dir,
                                                                                            base_checkpoints)
    new_probabilities, new_confidences, new_predictions = get_all_checkpoint_predictions(experiment_config, dataset,
                                                                                         dataset.test_dataloader(),
                                                                                         checkpoint_dir,
                                                                                         new_checkpoints)

    base_conf_scores = compute_conf_score(base_confidences)
    new_conf_scores = compute_conf_score(new_confidences)

    base_estimated_avg_conf_scores = compute_estimated_avg_conf_score(experiment_config, dataset, model_base,
                                                                      checkpoint_dir, base_checkpoints)
    new_estimated_avg_conf_scores = compute_estimated_avg_conf_score(experiment_config, dataset, model_new,
                                                                     checkpoint_dir, new_checkpoints)

    learned_combiner = combiners.create("stacking", base_model=model_base, new_model=model_new, dataset=dataset,
                                        stacker_name="lr")

    meta_conf_choices = (new_conf_scores > base_conf_scores).astype(int)
    meta_avg_conf_choices = (new_estimated_avg_conf_scores > base_estimated_avg_conf_scores).astype(int)
    meta_combined_choices = ((new_conf_scores > base_conf_scores) & (
                new_estimated_avg_conf_scores > base_estimated_avg_conf_scores)).astype(int)

    stacked_predictions = np.stack([base_final_predictions, new_final_predictions], axis=1)
    sample_indices = np.arange(len(stacked_predictions))

    meta_conf_predictions = stacked_predictions[sample_indices, meta_conf_choices]
    meta_avg_conf_predictions = stacked_predictions[sample_indices, meta_avg_conf_choices]
    meta_learned_predictions = learned_combiner.predict(dataset.test_dataloader())
    meta_combined_predictions = stacked_predictions[sample_indices, meta_combined_choices]

    all_predictions = {"conf": meta_conf_predictions, "avg_conf": meta_avg_conf_predictions,
                       "learned": meta_learned_predictions, "combined": meta_combined_predictions}

    nf_counts = {"count": [], "combiner": []}

    for combiner in all_predictions.keys():
        nfs = np.sum(
            ((base_final_predictions == dataset.test_labels) & (all_predictions[combiner] != dataset.test_labels)))
        nf_counts["count"].append(nfs)
        nf_counts["combiner"].append(combiner)

    pf_counts = {"count": [], "combiner": []}

    for combiner in all_predictions.keys():
        pfs = np.sum(
            ((base_final_predictions != dataset.test_labels) & (all_predictions[combiner] == dataset.test_labels)))
        pf_counts["count"].append(pfs)
        pf_counts["combiner"].append(combiner)

    nf_counts = pd.DataFrame(nf_counts)
    pf_counts = pd.DataFrame(pf_counts)

    cfg = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="stability")
    wandb_logger.experiment.config.update(cfg)

    all_predictions = {"sample_idx": list(np.arange(len(dataset.test_labels))) * 7, "prediction": np.concatenate(
        [base_final_predictions, new_final_predictions, meta_conf_predictions, meta_avg_conf_predictions, meta_learned_predictions,
         meta_combined_predictions, dataset.test_labels]),
                       "method": (["base"] * len(dataset.test_labels)) + (["new"] * len(dataset.test_labels)) + (
                                   ["conf"] * len(dataset.test_labels)) + (["avgconf"] * len(dataset.test_labels)) + (
                                             ["learned"] * len(dataset.test_labels)) + (
                                             ["combined"] * len(dataset.test_labels))
                       + ["label"] * len(dataset.test_labels)}

    all_predictions = pd.DataFrame(all_predictions)
    all_predictions.to_csv("predictions.csv")
    nf_counts.to_csv("nf_counts.csv")
    pf_counts.to_csv("pf_counts.csv")


if __name__ == "__main__":
    main()
