import logging
import os

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger

from settings import ROOT_DIR
from src.data import datasets
from src.gradient_projectors import gradient_projectors
from src.gradient_projectors.utils import get_per_sample_gradients, grad_vector_to_parameters
from src.losses import losses
from src.models import models
from src.utils.hydra import get_wandb_run

os.chdir(ROOT_DIR)
config_path = os.path.join(ROOT_DIR, "configs")
OmegaConf.register_new_resolver("wandb_run", get_wandb_run)


def evaluate_model(model, dataloader, device, loss_fn):
    correct = 0
    count = 0
    loss = 0

    with torch.no_grad():
        for x, y, _, _ in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, 1)
            prob = torch.nn.Softmax(dim=1)(out)
            sample_indices = torch.arange(len(prob))
            predicted_prob = prob[sample_indices, pred]
            correct += (pred == y).sum().item()
            count += len(x)
            loss += loss_fn(out, y).item() * len(x)

    acc = correct / float(count)
    loss = loss / float(count)

    return acc, loss, pred.detach().cpu().numpy(), predicted_prob.detach().cpu().numpy(), prob


def cosine_similarity(gradients, gradient):
    return np.dot(gradients, gradient) / (np.linalg.norm(gradients, axis=1, ord=2) * np.linalg.norm(gradient, ord=2))


def magnitude(gradient):
    return np.linalg.norm(gradient, ord=2)


@hydra.main(config_path=config_path, config_name="low_curvature")
def main(args: DictConfig):
    logging.info("\n" + OmegaConf.to_yaml(args))
    logging.info("Saving to: {}".format(os.getcwd()))

    seed_everything(args.misc.seed)

    dataset = datasets.create(args.data.name, **args.data.params)
    random_indices = np.random.choice(np.arange(len(dataset.train_data)), size=args.num_samples, replace=False)
    dataset.train_data.data = dataset.train_data.data[random_indices]
    dataset.train_data.targets = dataset.train_data.targets[random_indices]
    dataset.train_data.indices = dataset.train_data.indices[random_indices]
    dataloader = dataset.train_dataloader_inference(args.num_samples)

    model = models.create(args.model.name, num_classes=dataset.num_classes, num_channels=dataset.num_channels,
                          height=dataset.height, **args.model.params)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = losses.create(args.loss.name, **args.loss.params)
    device = "cuda:0"
    model = model.to(device)
    gradient_projector = gradient_projectors.create(args.gradient_projector.name, **args.gradient_projector.params)

    wandb.login(key="604640cf55056fd18bf07355ea2757e21a0c8d17")
    wandb_logger = WandbLogger(project="stability")
    cfg = OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )
    wandb_logger.experiment.config.update(cfg)

    prev_pred = None
    nfr = 0
    nfrs = {"nfr": [], "epoch": []}
    cosine_similarities = {"cosine_similarity": [], "epoch": [], "source": []}
    gradient_magnitudes = {"magnitude": [], "epoch": [], "source": []}
    per_sample_gradient_magnitudes = {"sample_idx": [], "epoch": [], "magnitude": []}
    predictions = {"sample_idx": [], "epoch": [], "pred": [], "prob": []}

    for epoch in range(1, args.epochs + 1):
        for x, y, _, _ in dataloader:
            x, y = x.to(device), y.to(device)

            projected_gradient = gradient_projector.project(model, x, y, loss_fn)
            grad_vector_to_parameters(projected_gradient.to(x.device), model.parameters())

            per_sample_gradients = get_per_sample_gradients(model, x, y, loss_fn).detach().cpu().numpy()
            overall_gradient = np.mean(per_sample_gradients, axis=0)
            original_cosine_similarities = cosine_similarity(per_sample_gradients, overall_gradient)
            projected_cosine_similarities = cosine_similarity(per_sample_gradients, projected_gradient)
            original_gradient_magnitude = magnitude(overall_gradient)
            projected_gradient_magnitude = magnitude(projected_gradient)
            epoch_per_sample_magnitudes = [magnitude(per_sample_gradients[i]) for i in range(len(per_sample_gradients))]

            cosine_similarities["cosine_similarity"] += list(original_cosine_similarities)
            cosine_similarities["cosine_similarity"] += list(projected_cosine_similarities)
            cosine_similarities["epoch"] += [epoch] * len(x) * 2
            cosine_similarities["source"] += ["original"] * len(x)
            cosine_similarities["source"] += ["projected"] * len(x)

            gradient_magnitudes["magnitude"] += [original_gradient_magnitude, projected_gradient_magnitude]
            gradient_magnitudes["epoch"] += [epoch] * 2
            gradient_magnitudes["source"] += ["original", "projected"]

            per_sample_gradient_magnitudes["sample_idx"] += list(range(len(x)))
            per_sample_gradient_magnitudes["epoch"] += [epoch] * len(x)
            per_sample_gradient_magnitudes["magnitude"] += epoch_per_sample_magnitudes

            optimizer.step()
            optimizer.zero_grad()
            acc, loss, pred, prob, _ = evaluate_model(model, dataloader, device, loss_fn)

            predictions["sample_idx"] += list(range(len(x)))
            predictions["epoch"] += [epoch] * len(x)
            predictions["pred"] += list(pred)
            predictions["prob"] += list(prob)

            wandb_logger.log_metrics({"train/loss": loss}, step=epoch)
            wandb_logger.log_metrics({"train/acc": acc}, step=epoch)

        if prev_pred is not None:
            nfr = np.mean((prev_pred == y.detach().cpu().numpy()) & (pred != prev_pred))
            nfrs["nfr"].append(nfr)
            nfrs["epoch"].append(epoch)

        if epoch % args.log_frequency == 0:
            result_path = "checkpoints"

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            torch.save(model.state_dict(), result_path + "/checkpoint_{}.pt".format(epoch))

        prev_pred = pred
        wandb_logger.log_metrics({"train/nfr": nfr}, step=epoch)

    cosine_similarities = pd.DataFrame(cosine_similarities)
    cosine_similarities.to_csv("cosine_similarities.csv")
    gradient_magnitudes = pd.DataFrame(gradient_magnitudes)
    gradient_magnitudes.to_csv("gradient_magnitudes.csv")
    per_sample_gradient_magnitudes = pd.DataFrame(per_sample_gradient_magnitudes)
    per_sample_gradient_magnitudes.to_csv("per_sample_gradient_magnitudes.csv")
    predictions = pd.DataFrame(predictions)
    predictions.to_csv("predictions.csv")
    nfrs = pd.DataFrame(nfrs)
    nfrs.to_csv("nfrs.csv")


if __name__ == "__main__":
    main()
