import torch
from pytorch_lightning import Trainer

from src.lightning_modules import lightning_modules


def extract_predictions(dataloader, model):
    module = lightning_modules.create("prediction", model=model)
    trainer = Trainer(gpus=1, enable_checkpointing=False, enable_progress_bar=False)

    logits = trainer.predict(module, dataloaders=dataloader)
    logits = torch.cat(logits)
    preds = torch.argmax(logits, dim=1)

    return preds.detach().cpu().numpy()