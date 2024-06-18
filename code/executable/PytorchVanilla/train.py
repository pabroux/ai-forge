import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import getLogger, chooseDevice
from utils_train import (
    check_given_arg_nb,
    secure_existing_model,
    tensorboard_add_model,
    train_one_epoch,
    validation_one_epoch,
    EarlyStopping,
)


if __name__ == "__main__":

    # Checking the number of given arguments
    check_given_arg_nb()

    # Importing the variables from the config file
    config = __import__(
        sys.argv[1].replace(".py", "").lstrip("./").replace("/", "."), fromlist=[""]
    )
    path_model_to_save = config.path_model_to_save
    path_tensorboard_to_save = config.path_tensorboard_to_save
    verbose = config.verbose
    security = config.security
    model = config.model
    dataset = config.dataset
    trainset_root = config.trainset_root
    trainloader_shuffle = config.trainloader_shuffle
    trainloader_num_workers = config.trainloader_num_workers
    devset_root = config.devset_root
    devloader_shuffle = config.devloader_shuffle
    devloader_num_workers = config.devloader_num_workers
    preprocessor = config.preprocessor
    preprocessor_target = config.preprocessor_target
    loss = config.loss
    metrics = config.metrics
    optimizer = config.optimizer
    earlystopping = config.earlystopping
    earlystopping_min_delta = config.earlystopping_min_delta
    earlystopping_patience = config.earlystopping_patience
    earlystopping_restore_best_weights = config.earlystopping_restore_best_weights
    epochs = config.epochs
    batch_size = config.batch_size
    device = config.device.lower()

    # Initializing logger
    logger = getLogger()

    # Checking if a model is already saved at the given path
    secure_existing_model(security, path_model_to_save, logger)

    # Initializing tensorboard
    writer = SummaryWriter(path_tensorboard_to_save)

    # Choosing the device to use with the network
    torch_device = chooseDevice(device, logger)

    # Loading the train and dev set
    trainset = dataset(
        root=trainset_root,
        train=True,
        transform=preprocessor,
        target_transform=preprocessor_target,
    )
    devset = dataset(
        root=devset_root,
        train=False,
        transform=preprocessor,
        target_transform=preprocessor_target,
    )

    # Setting the train and dev loader
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=trainloader_shuffle,
        num_workers=trainloader_num_workers,
    )

    devloader = torch.utils.data.DataLoader(
        devset,
        batch_size=batch_size,
        shuffle=devloader_shuffle,
        num_workers=devloader_num_workers,
    )

    # Configure model
    model = model()

    # Configure optimizer
    optimizer = optimizer(model)

    # Adding the model to tensorboard
    tensorboard_add_model(model, writer, trainset)

    # Transferring the model into the selected device
    model = model.to(torch_device)

    # Initializing the EarlyStopping
    if earlystopping:
        early_stopping = EarlyStopping(
            model,
            optimizer,
            patience=earlystopping_patience,
            min_delta=earlystopping_min_delta,
            restore_best_weights=earlystopping_restore_best_weights,
            logger=logger,
        )

    # Optimization loop
    for epoch in range(epochs):

        # Train loop (Train one epoch)
        running_train_loss, running_train_metrics = train_one_epoch(
            model,
            epoch,
            epochs,
            trainloader,
            torch_device,
            optimizer,
            loss,
            metrics,
            writer,
        )

        # Validation loop (Validation one epoch)
        running_dev_loss, running_dev_metrics = validation_one_epoch(
            model,
            epoch,
            epochs,
            devloader,
            torch_device,
            loss,
            metrics,
            writer,
        )

        # Logging training loss and validation loss into console (verbose)
        if verbose:
            logger.info(
                f"Loss/Training: ~{running_train_loss / len(trainloader):.5f} | Loss/Validation: {running_dev_loss / len(devloader):.5f}"
            )
            for name in metrics:
                logger.info(
                    f"Metric/Training_{name}: {running_train_metrics[name] / len(trainloader):.5f} | Metric/Validation_{name}: {running_dev_metrics[name] / len(devloader):.5f}"
                )

        # Assessing EarlyStopping
        if earlystopping:
            early_stopping(running_dev_loss / len(devloader))
            if early_stopping.early_stop:
                break

    # Saving the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path_model_to_save,
    )

    # Closing tensorboard
    writer.close()

    logger.info("Training done 👍")
