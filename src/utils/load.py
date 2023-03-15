import os
from collections import OrderedDict

import torch

from src.models import models


def remove_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "original_" in k:
            continue

        name = k.replace("model.", "")
        new_state_dict[name] = v

    return new_state_dict


def load_model(base_dir, checkpoint, args, dataset):
    checkpoint = torch.load(os.path.join(base_dir, checkpoint))
    state_dict = remove_prefix(checkpoint["state_dict"])

    model = models.create(args.model.name, num_classes=dataset.num_classes, **dataset.stats, **args.model.params)
    model.load_state_dict(state_dict)

    return model


def get_checkpoints(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoints_dir, f)))

    return checkpoints


def find_last_checkpoint_for_update(checkpoints, update_num):
    relevant_checkpoints = [checkpoint for checkpoint in checkpoints if
                            "update_num={}.ckpt".format(update_num) in checkpoint]

    return relevant_checkpoints[-1]