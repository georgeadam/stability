# Project Description

This repo contains code for running experiments that evaluate the predictive churn of various churn reduction 
baseline methods, as well as the novel method accumulated model combination (AMC). Churn is defined as new errors 
introduced by an updated model version relative to a base model, and can occur even when the new model is more accurate
on average. Details about baseline methods and AMC can be found in the [paper](https://arxiv.org/pdf/2305.04135.pdf).

# Repo Structure

The structure of this repo is 
* `configs` (has all the .yaml files which configure model training)
* `src` (our code files which define models, datasets, imputers, etc.)

In order to run the code another directory called `data` needs to be created, which houses the datasets used.

## src

`src/experiments` contains the scripts which train models and evaluate the predictive churn of methods such as distillation,
warm start, ensembles, and AMC. Experiments are configured using [hydra](https://hydra.cc/docs/intro/) which enables the use of
the factory design pattern when creating different models, datasets, etc. For example, the `src/models` folder defines 
the model architectures used in the experiments, and the `src/models/creation.py` file defines the `models` factory which
is used to create the model objects as follows:

`model = models.create(args.model.name, num_classes=dataset.num_classes, **dataset.stats, **args.model.params)`

This avoids the need for unwieldy if/else blocks, and allows for models that take in arbitrary parameters to be created
in a standardized way. For example, `src/models/fc.py` defines a single hidden layer fully connected architecture for
tabular data which requires arguments `num_classes` and `num_features` to create. The `src/models/mobilenet.py` model
wraps the Torchvision implementatation of the MobileNet architecture and is created using the arguments `num_classes`,
`num_channels`, and `height`. Both of these models can be created conveniently via

`model = models.create(args.model.name, num_classes=dataset.num_classes, **dataset.stats, **args.model.params)`

Another fundamental object required for running experiments is the Pytorch Lightning DataModule. `src/data/data_module`
extends the original Lightning DataModule with additional functionality and properties. There are additional DataLoaders 
defined for performing inference, and there are data transforms defined here as well. The DataModule encapsulates the 
training data, validation data, test data, and extra data (used for updating the model) in one centralized location. All 
datasets extend this DataModule class, and override the `setup()` method to define how the data should be loaded and processed.
This allows for image, text, and tabular datasets to all be used with one standardized interface, so the same training script
can be applied to arbitrary model/dataset combinations.

Model training is defined in `src/lightning_modules` which houses LightningModule objects that define the loss function
to be used and how performance metrics are logged. 

Below are brief descriptions of the remaining submodules:

* `src/annealers` defines pacing functions used in curriculum learning
* `src/base` defines the generic Factory class used for constructing objects
* `src/callbacks` defines callbacks for things like custom early stopping, scoring samples, and tracking predictions throughout training
* `src/combiners` defines ways of combining the predictions of base and new model used by AMC
* `src/ensemblers` defines ways of ensembling models for the ensemble baseline
* `src/gradient_projectors` defines ways of manipulating the standard gradient to achieve self-consistent learning between epochs
* `src/inferers` is used for performing inference using LightningModule, and avoids the need for defining a prediction loop
* `src/label_smoothers` defines ways of performing label smoothing, another churn reduction baseline
* `src/losses` defines some custom loss functions
* `src/lr_schedulers` wraps existing PyTorch learning rate schedulers so that creating them has a standardized interface
* `src/optimizers` wraps existing PyTorch optimizers so that creating them has a standardized interface
* `src/samplers` defines sampling functions used in curriculum learning
* `src/trainers` defines custom trainers
* `src/utils` defines utility functions for loading in models and logging metrics to WandB

# Libraries

To install the necessary libraries, run the command

`pip install -r requirements.txt`

# Model Training

Much of the training code depends on [wandb](https://wandb.ai/site), so in order to run any experiments, one has to locally setup their 
wandb account such that they can login with their API key by entering their key in `src/utils/wandb.py`. Additionally, 
 users should have a project named `stability` created in their wandb account. 

Add the path to the main directory (one level above `src`) to the `PYTHONPATH` via

`export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

To train the baseline without any regularization, one needs to call 

`python src/scripts/run_cold_start.py`

which will use the default arguments from `configs/cold_start.yaml`. To specify custom arguments,
i.e. change the dataset or model, these parameters can be specfied at the command line via

`python src/scripts/run_cold_start.py model=resnet50 data=cifar10`