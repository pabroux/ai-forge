import os
import sys
import typing
import logging
import copy
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import get_logger


def check_given_arg_nb() -> None:
    """Checking the number of given arguments

    Args:
        None

    Returns:
        None

    """
    if len(sys.argv) != 2:
        sys.exit(
            f"Error: Wrong number of given arguments.\nHere how you should use it: python {sys.argv[0]} <path_to_the_python_config_file>"
        )
    if not os.path.exists(sys.argv[1]):
        sys.exit(f"Error: No file found at the given config file path.")


def secure_existing_model(
    security: bool, path_model_to_save: str, logger: logging.Logger = None
) -> None:
    """Checking if a model is already saved at the given path and aborting the training according to the "security" parameter.

    Args:
        security (bool): activating or disabling the security. If true, aborts the training if a model is already saved at "path_model_to_save".
        path_model_to_save (str): path to where we want to save the model
        logger (logging.Logger): a specific logger to use

    Returns:
        None

    """
    if logger is None:
        logger = get_logger()
    if security and os.path.exists(path_model_to_save):
        logger.warning(
            "A model is already saved at the given path. Should I go ahead? (Y/n)"
        )
        while (answer := input()) not in ["Y", "n"]:
            logger.warning("Please, enter 'Y' for yes or 'n' for no.")
        if answer == "n":
            sys.exit()
    elif not security:
        logger.warning("Security is OFF. An existing model could be overwritten")


def tensorboard_add_model(
    model: torch.nn.Module, writer: SummaryWriter, dataset: torch.utils.data.Dataset
) -> None:
    """Adding the model to tensorboard

    Args:
        model (torch.nn.Module): model we would like to add to tensorboard
        writer (torch.utils.tensorboard.SummaryWriter): a SummaryWriter instance (tensorboard) on which we want to write
        dataset (torch.utils.data.Dataset): any dataset which the model can use

    Returns:
        None

    """
    dataiter = iter(torch.utils.data.DataLoader(dataset))
    X, _ = dataiter.next()
    writer.add_graph(model, X)


def train_one_epoch(
    model: torch.nn.Module,
    epoch: int,
    epochs: int,
    trainloader: DataLoader,
    torch_device: torch.device,
    optimizer: Optimizer,
    loss: typing.Any,
    metrics: typing.Dict[str, typing.Any],
    writer: SummaryWriter,
) -> typing.Tuple[float, typing.Dict[str, typing.Any]]:
    """Training one epoch

    Args:
        model (torch.nn.Module): model we would like to use
        epoch (int): the current epoch
        epochs (int): the number of epochs
        trainloader (torch.utils.data.DataLoader): the train dataloader
        torch_device (torch.device): the torch.device object on which we want to train
        optimizer (torch.optim.Optimizer): the optimizer for updating the weights
        loss (typing.Any): a method or an object that can take "(y_pred,y_target)" as parameters
        metrics (typing.Dict[str, typing.Any]): the metrics we want to apply
        writer (torch.utils.tensorboard.SummaryWriter): a SummaryWriter instance (tensorboard) on which we want to write

    Returns:
       tuple(float,typing.Dict[str, typing.Any]): a tuple containing the running train loss and the running train metrics

    """

    running_train_loss = 0.0
    running_train_metrics = {name: 0 for name in metrics}

    for trainbatch_data in tqdm(
        trainloader,
        colour="#BF004C",
        bar_format=f"EPOCH {epoch + 1}/{epochs} - TRAINING  "
        + "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}",
    ):
        # Getting the inputs and labels
        X_train, y_train = trainbatch_data

        # Transferring the data into the selected device
        X_train, y_train = X_train.to(torch_device), y_train.to(torch_device)

        # Setting the parameter gradients to zero
        optimizer.zero_grad()

        # Applying the forward pass
        y_pred = model(X_train)

        # Loss assessing
        loss_model = loss(y_pred, y_train)

        # Applying the backward pass (Gradient assessing)
        loss_model.backward()

        # Updating the weigths
        optimizer.step()

        # Preparing the training loss log
        running_train_loss += loss_model.item()

        # Preparing the training metric logs
        with torch.no_grad():
            running_train_metrics = {
                name: running_train_metrics[name] + fn(y_pred, y_train)
                for name, fn in metrics.items()
            }

    # Logging training step into tensorboard
    ## It's an average of the batch losses during training. It gives an estimate of the "epoch loss"
    writer.add_scalar("Loss/Training", running_train_loss / len(trainloader), epoch)
    ## It's an average of the batch metrics during training. It gives an estimate of each "epoch metric"
    for name in running_train_metrics:
        writer.add_scalar(
            "Metric/Training_" + name,
            running_train_metrics[name] / len(trainloader),
            epoch,
        )

    return (running_train_loss, running_train_metrics)


def validation_one_epoch(
    model: torch.nn.Module,
    epoch: int,
    epochs: int,
    devloader: DataLoader,
    torch_device: torch.device,
    loss: typing.Any,
    metrics: typing.Dict[str, typing.Any],
    writer: SummaryWriter,
) -> typing.Tuple[float, typing.Dict[str, typing.Any]]:
    """Validating one epoch

    Args:
        model (torch.nn.Module): model we would like to use
        epoch (int): the current epoch
        epochs (int): the number of epochs
        devloader (torch.utils.data.DataLoader): the dev dataloader
        torch_device (torch.device): the torch.device object on which we want to train
        loss (typing.Any): a method or an object that can take "(y_pred,y_target)" as parameters
        metrics (typing.Dict[str, typing.Any]): the metrics we want to apply
        writer (torch.utils.tensorboard.SummaryWriter): a SummaryWriter instance (tensorboard) on which we want to write

    Returns:
       tuple(float,typing.Dict[str, typing.Any]): a tuple containing the running dev loss and the running dev metrics

    """

    running_dev_loss = 0.0
    running_dev_metrics = {name: 0 for name in metrics}

    with torch.no_grad():
        for devbatch_data in tqdm(
            devloader,
            colour="green",
            bar_format=f"EPOCH {epoch + 1}/{epochs} - VALIDATION"
            + "{desc:<5.5}{percentage:3.0f}%|{bar:50}{r_bar}",
        ):
            # Getting the inputs and labels
            X_dev, y_dev = devbatch_data

            # Transferring the data into the selected device
            X_dev, y_dev = X_dev.to(torch_device), y_dev.to(torch_device)

            # Applying the forward pass
            y_pred = model(X_dev)

            # Loss assessing
            loss_model = loss(y_pred, y_dev)

            # Preparing the validation loss log
            running_dev_loss += loss_model.item()

            # Preparing the validation metric log
            running_dev_metrics = {
                name: running_dev_metrics[name] + fn(y_pred, y_dev)
                for name, fn in metrics.items()
            }

    # Logging validation step into tensorboard
    writer.add_scalar("Loss/Validation", running_dev_loss / len(devloader), epoch)
    for name in running_dev_metrics:
        writer.add_scalar(
            "Metric/Validation_" + name,
            running_dev_metrics[name] / len(devloader),
            epoch,
        )

    return (running_dev_loss, running_dev_metrics)


class EarlyStopping:
    """
    EarlyStopping
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        patience: int = 5,
        min_delta: float = 0,
        restore_best_weights: bool = True,
        logger: logging.Logger = None,
    ) -> None:
        """Initializing the earlyStopping

        Args:
            model (torch.nn.Module): model on which to apply the early stopping
            optimizer (torch.optim.Optimizer): optimizer on which to apply the early stopping
            partience (int): how many epochs to wait before stopping
            min_delta (float): minimium amount of change to count as an improvement
            restore_best_weights (bool): restoring or not the best weights
            logger (logging.Logger): a specific logger to use

        Returns:
            None

        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.loss_best = None
        self.early_stop = False
        self.model = model
        self.model_best_state_dict = None
        self.optimizer = optimizer
        self.optimizer_best_state_dict = None
        self.restore_best_weights = restore_best_weights
        self.logger = get_logger() if logger is None else logger

    def __call__(self, loss: float) -> None:
        """Assessing the earlyStopping
        If earlyStopping is triggered, self.early_stop is set to True

        Args:
            loss (float): model on which to apply the early stopping

        Returns:
            None

        """
        if not self.early_stop:
            if self.loss_best is None or (self.loss_best - loss) >= self.min_delta:
                self.counter = 0
                self.loss_best = loss  # type: ignore
                self.model_best_state_dict = copy.deepcopy(self.model.state_dict())  # type: ignore
                self.optimizer_best_state_dict = copy.deepcopy(  # type: ignore
                    self.optimizer.state_dict()
                )
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                    self.logger.info("EarlyStopping triggered")
                    if self.restore_best_weights:
                        self.restore_best_params()

    def restore_best_params(self) -> None:
        """Restoring the best found weigths

        Args:
            None

        Returns:
            None

        """
        self.model.load_state_dict(self.model_best_state_dict)  # type:ignore
        self.optimizer.load_state_dict(self.optimizer_best_state_dict)  # type:ignore
