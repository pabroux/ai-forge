import importlib
import importlib.util
import lightning.pytorch as pl
import os
import sys
from utils import get_logger, CustomProgressBar


if __name__ == "__main__":

    # AUTOMATIC (NON-CONFIGURABLE)
    ## Checking the given parameters
    if len(sys.argv) != 3:
        raise Exception("Syntax: python test.py config_file_path checkpoint_file_path")

    ## Configuration file to load
    config_path = os.path.realpath(sys.argv[1])

    ## Checkpoint file to load
    checkpoint_path = os.path.realpath(sys.argv[2])

    ## Setting the working directory to the directory of the script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    ## Loading a custom logger
    log = get_logger()

    ## Loading a custom progress bar
    custom_bar = CustomProgressBar()

    ## Loading the configuration file
    log.info(f"Configuration file chosen: {config_path}")
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(config_spec)
    sys.modules["config"] = config
    config_spec.loader.exec_module(config)

    ## Setting up the model, the datamodule, the logger and the trainer from the configuration file
    model = config.model.load_from_checkpoint(checkpoint_path)
    datamodule = config.datamodule
    trainer = pl.Trainer(
        accelerator=config.trainer__accelerator,
        devices=config.trainer__devices,
        strategy=config.trainer__strategy,
        num_nodes=config.trainer__num_nodes,
        sync_batchnorm=config.trainer__sync_batchnorm,
        logger=None,
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

    ## Testing
    log.info("Testing: START")
    trainer.test(model, datamodule=datamodule)
    log.info("Testing: STOP")
