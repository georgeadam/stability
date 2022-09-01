import numpy as np
import torch
import wandb
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters, _check_param_device
from torchmetrics.functional import accuracy
from tqdm.auto import tqdm

from .creation import lightning_modules


class OGD(LightningModule):
    def __init__(self, model, original_model, optimizer, lr_scheduler, projection, num_samples):
        super().__init__()

        self.model = model
        self.original_model = original_model
        self.projection = projection
        self.num_samples = num_samples
        self.loss = torch.nn.CrossEntropyLoss()
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self.training_outputs = None
        self.validation_outputs = None

        self.ogd_basis = None

    def forward(self, batch):
        # Used only by trainer.predict() to evaluate the model's predictions
        x, y, idx, source = batch

        logits = self.model(x)

        return logits

    def training_step(self, batch, batch_idx):
        metrics = self._get_all_metrics(batch)

        # Log loss and metric
        self.log('train/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('train/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('train/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def training_epoch_end(self, outputs):
        outputs = self._stack_outputs(outputs)
        self.training_outputs = outputs

        # del self.ogd_basis
        # garbage_collection_cuda()
        # self.update_ogd_basis()

    def validation_epoch_end(self, outputs):
        outputs = self._stack_outputs(outputs)
        self.validation_outputs = outputs

    def _stack_outputs(self, outputs):
        names = list(outputs[0].keys())
        stacked_outputs = {k: [] for k in names}

        for out in outputs:
            for k in names:
                if isinstance(out[k], torch.Tensor):
                    if len(out[k].shape) == 0:
                        stacked_outputs[k].append(out[k].unsqueeze(0).cpu().numpy())
                    else:
                        stacked_outputs[k].append(out[k].cpu().numpy())
                elif isinstance(out[k], float):
                    stacked_outputs[k].append(np.array([out[k]]))
                else:
                    stacked_outputs[k].append(out[k])

        stacked_outputs = {k: np.concatenate(v) for k, v in stacked_outputs.items()}

        return stacked_outputs

    def validation_step(self, batch, batch_idx):
        if self.global_step == 0:
            wandb.define_metric('val/accuracy', summary='max')

        metrics = self._get_all_metrics(batch)

        # Log loss and metric
        self.log('val/loss', metrics["loss"], on_step=False, on_epoch=True)
        self.log('val/accuracy', metrics["acc"], on_step=False, on_epoch=True)
        self.log('val/epoch', self.current_epoch, on_step=False, on_epoch=True)

        return metrics

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                         lr=self._optimizer.params.lr,
                                         momentum=0,
                                         weight_decay=0)
        return self.optimizer

    def get_params_dict(self):
        return self.model.parameters()

    def _get_all_metrics(self, batch):
        x, y, index, source = batch
        logits = self.model(x)
        loss = self._get_loss(logits, y)

        with torch.no_grad():
            stats = self._get_stats(x, index, source, logits, y)

        stats["loss"] = loss

        return stats

    def _get_loss(self, logits, y):
        return self.loss(logits, y)

    def _get_stats(self, x, index, source, logits, y):
        preds = torch.argmax(logits, dim=1)
        probs = torch.nn.Softmax(dim=1)(logits)
        probs = probs[torch.arange(len(probs)), preds]

        if len(y.shape) > 1:
            y = torch.argmax(y, dim=1)

        acc = accuracy(preds, y)
        correct = preds == y

        if self.original_model:
            original_logits = self.original_model(x)
            original_preds = torch.argmax(original_logits, dim=1)
        else:
            original_preds = preds

        epoch = np.array([self.current_epoch] * len(preds))

        return {"preds": preds.cpu().numpy(), "y": y.cpu().numpy(), "correct": correct.cpu().numpy(),
                "index": index.cpu().numpy(), "epoch": epoch,
                "acc": acc, "original_preds": original_preds.cpu().numpy(), "source": source.cpu().numpy(),
                "probs": probs.cpu().numpy()}

    def _get_new_ogd_basis(self):
        return self._get_neural_tangents()

    def _get_neural_tangents(self):
        new_basis = []
        train_dataset = self.trainer.train_dataloader.dataset.datasets
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        for i, (x, y, index, source) in tqdm(enumerate(train_dataloader),
                                             desc="get neural tangents",
                                             total=len(train_dataloader.dataset)):
            # only care about matching samples from the original training data
            if source != 0:
                continue

            # if gpu:
            x = self.to_device(x)
            y = self.to_device(y)

            with torch.no_grad():
                orig_out = self.original_model(x)
                orig_pred = torch.argmax(orig_out, dim=1)

            # only want to match samples correctly predicted by the original model
            if orig_pred != y:
                continue

            out = self.model(x)
            out = torch.nn.Softmax(dim=1)(out)
            label = y.item()
            pred = out[0, label]

            self.optimizer.zero_grad()
            pred.backward()

            grad_vec = parameters_to_grad_vector(self.get_params_dict())

            if self.projection == "other_way":
                new_basis.append(-grad_vec)
            else:
                new_basis.append(grad_vec)

            if len(new_basis) == self.num_samples:
                break

        new_basis_tensor = torch.stack(new_basis).T
        del new_basis
        garbage_collection_cuda()

        return new_basis_tensor.detach()

    def to_device(self, tensor):
        if self.trainer.gpus:
            return tensor.cuda()
        else:
            return tensor

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx=0,
                       optimizer_closure=None,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False,
                       ):

        optimizer_closure()
        cur_param = parameters_to_vector(self.get_params_dict())
        grad_vec = parameters_to_grad_vector(self.get_params_dict())

        optimizer.zero_grad()
        if self.projection == "orthogonal":
            self.update_ogd_basis()

            proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis, gpu=self.trainer.gpus)
            del self.ogd_basis
            garbage_collection_cuda()
            new_grad_vec = grad_vec - proj_grad_vec

            cur_param -= self._optimizer.params.lr * new_grad_vec
        elif self.projection == "same" or self.projection == "other_way":
            self.update_ogd_basis()

            proj_grad_vec = project_vec(grad_vec, proj_basis=self.ogd_basis, gpu=self.trainer.gpus)
            del self.ogd_basis
            garbage_collection_cuda()
            new_grad_vec = proj_grad_vec

            cur_param -= self._optimizer.params.lr * new_grad_vec
        else:
            cur_param -= self._optimizer.params.lr * grad_vec

        vector_to_parameters(cur_param, self.get_params_dict())

        optimizer.zero_grad()

    def _update_mem(self):
        # (e) Get the new non-orthonormal gradients basis
        # Non orthonormalised basis
        new_basis_tensor = self._get_new_ogd_basis()
        print(f"new_basis_tensor shape {new_basis_tensor.shape}")

        # (f) Ortonormalise the whole memorized basis
        self.ogd_basis = new_basis_tensor
        self.ogd_basis = orthonormalize(self.ogd_basis, gpu=self.trainer.gpus, normalize=True)

    def update_ogd_basis(self):
        device = torch.device("cuda")
        self.model.to(device)
        print(f"\nself.model.device update_ogd_basis {next(self.model.parameters()).device}")
        self._update_mem()


lightning_modules.register_builder("ogd", OGD)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def parameters_to_grad_vector(parameters):
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def grad_vector_to_parameters(vec, parameters):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param


def project_vec(vec, proj_basis, gpu):
    if proj_basis.shape[1] > 0:  # param x basis_size
        dots = torch.matmul(vec, proj_basis)  # basis_size
        # out = torch.matmul(proj_basis, dots)
        # TODO : Check !!!!
        out = torch.matmul(proj_basis, dots.T)
        return out
    else:
        return torch.zeros_like(vec)


def orthonormalize(vectors, gpu, normalize=True, start_idx=0):
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
    # TODO : Check if start_idx is correct :)
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize:
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]

    if start_idx == 0:
        start_idx = 1
    for i in tqdm(range(start_idx, vectors.size(1)), desc="orthonormalizing ..."):
        vector = vectors[:, i]
        V = vectors[:, :i]
        PV_vector = torch.mv(V, torch.mv(V.t(), vector))
        if normalize:
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors
