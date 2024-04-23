import importlib
import importlib.util
import lightning.pytorch as pl
import mlflow
import os
import sys
from mlflow.models.signature import infer_signature
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.tuner import Tuner
from utils import getLogger, CustomProgressBar


if __name__ == "__main__":

    # AUTOMATIC (NON-CONFIGURABLE)
    ## Configuration file to load
    config_path = (
        os.path.dirname(os.path.realpath(__file__)) + "/config/config_test.py"
        if len(sys.argv) == 1
        else os.path.realpath(sys.argv[1])
    )

    ## Setting the working directory to the directory of the script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    ## Loading a custom logger
    log = getLogger()

    ## Loading a custom progress bar
    custom_bar = CustomProgressBar()

    ## Loading the configuration file
    log.info(f"Configuration file chosen: {config_path}")
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(config_spec)
    sys.modules["config"] = config
    config_spec.loader.exec_module(config)

    ## Setting up the seed
    if config.extra_settings__seed:
        pl.seed_everything(
            config.extra_settings__seed_value,
            workers=config.extra_settings__seed_workers,
        )
        ### Reloading the configuration file
        config_spec.loader.exec_module(config)

    ## Setting up the model, the datamodule, the logger and the trainer from the configuration file
    model = config.model
    datamodule = config.datamodule
    logger = config.logger
    trainer = pl.Trainer(
        accelerator=config.trainer__accelerator,
        devices=config.trainer__devices,
        strategy=config.trainer__strategy,
        num_nodes=config.trainer__num_nodes,
        sync_batchnorm=config.trainer__sync_batchnorm,
        logger=logger,
        min_epochs=config.trainer__min_epochs,
        max_epochs=config.trainer__max_epochs,
        precision=config.trainer__precision,
        gradient_clip_val=config.trainer__gradient_clip_val,
        reload_dataloaders_every_n_epochs=config.trainer__reload_dataloaders_every_n_epochs,
        callbacks=[custom_bar] + config.callbacks,
        enable_model_summary=config.trainer__enable_model_summary,
        profiler=config.trainer__profiler,
        check_val_every_n_epoch=config.trainer__check_val_every_n_epoch,
        val_check_interval=config.trainer__val_check_interval,
        num_sanity_val_steps=config.trainer__num_sanity_val_steps,
        benchmark=config.trainer__benchmark,
        deterministic=config.trainer__deterministic,
        fast_dev_run=config.trainer__fast_dev_run,
        overfit_batches=config.trainer__overfit_batches,
        enable_checkpointing=config.trainer__enable_checkpointing,
        default_root_dir=config.trainer__default_root_dir,
    )

    ## Setting up MLflow and activating it if it is chosen as logger
    if isinstance(logger, MLFlowLogger):
        mlflow.set_tracking_uri(logger._tracking_uri)
        mlflow.set_experiment(logger._experiment_name)
        if config.logger__log_system_metrics:
            mlflow.enable_system_metrics_logging()
        mlflow.start_run()
        logger._run_id = mlflow.active_run().info.run_id

        ## Saving the config file
        if config.logger__log_config:
            mlflow.log_artifact(config_path, artifact_path="config")

    ## Applying the model weight initializer
    if config.model__weight_initializer is not None:
        config.model__weight_initializer(model)
        logger.log_hyperparams(
            {"model_weight_initializer": config.model__weight_initializer.__name__}
        )

    ## Tuning (determining best batch size and/or suggested learning rate)
    if config.tuner__auto_scale_batch_size or config.tuner__auto_lr_find:
        log.info("Tuning model: START")
        log.info(
            "⮑  Auto scale batch size: "
            + str(bool(config.tuner__auto_scale_batch_size))
        )
        log.info("⮑  LR to investigate: " + str(bool(config.tuner__auto_lr_find)))
        tuner = Tuner(trainer)
        if config.tuner__auto_scale_batch_size:
            tuner.scale_batch_size(
                model,
                mode=config.tuner__auto_scale_batch_size_mode,
                datamodule=datamodule,
            )
            log.info("Best batch size: " + str(datamodule.batch_size))  # type:ignore
        if config.tuner__auto_lr_find:
            tuner.lr_find(
                model,
                datamodule=datamodule,
            )
        log.info("Tuning model: STOP")

    ## Training
    log.info("Training: START")
    trainer.fit(model, datamodule=datamodule)
    log.info("Training: STOP")

    ## Testing
    if config.extra_settings__test:
        if datamodule.test_dataloader() is None:
            log.warning("Testing: SKIP")
            log.warning(
                "⮑  The test loop can't be done because there is no test dataloader associated to the datamodule."
            )
        else:
            log.info("Testing: START")
            trainer.test(model, datamodule=datamodule)
            log.info("Testing: STOP")

    ## Desactivating MLflow if it is chosen as logger
    if isinstance(logger, MLFlowLogger):
        ### Saving the input size
        if config.logger__log_dataset:
            x, _ = next(iter(datamodule.train_dataloader()))
            dataset = mlflow.data.from_numpy(
                x.numpy(),
                name=datamodule.__class__.__name__,
            )
            mlflow.log_input(
                dataset,
                (
                    "train, dev & test"
                    if config.extra_settings__test
                    and datamodule.test_dataloader() is not None
                    else "train & dev"
                ),
            )

        ### Saving the model through the logger
        if config.logger__log_model:
            x, y = next(iter(datamodule.train_dataloader()))
            signature = infer_signature(x.numpy(), y.numpy())
            mlflow.pytorch.log_model(model, "model", signature=signature)
        mlflow.end_run()
