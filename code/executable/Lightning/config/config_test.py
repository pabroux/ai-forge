from datamodule import TestDataModule
from model import TestNet
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger

"""
Config Test

    description:
        This is a test configuration file
"""

"""
Extra settings

    variables:
        extra_settings__seed: Setting seeds for pseudo-random generators
        extra_settings__seed_value: The value to use as a seed
        extra_settings__seed_workers: Deriving unique seeds across all dataloader workers and processes for torch, numpy and stdlib random number generators
        extra_settings__test: Applying the test loop after the training
"""
extra_settings__seed = True
extra_settings__seed_value = 42
extra_settings__seed_workers = True
extra_settings__test = False

"""
Callbacks

    description:
        List of callbacks (EarlyStopping, ModelCheckpoint, GradientAccumulationScheduler, etc.)
"""
callbacks = [
    EarlyStopping(
        monitor="loss/val_loss",
        patience=3,
        strict=True,
        verbose=False,
        mode="min",
    ),
    ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="epoch={epoch}-val_loss={loss/val_loss:.2f}",
        auto_insert_metric_name=False,
    ),
]

"""
Model

    variables:
        model: The model to use
        model__weight_initializer: The model weight initializer to apply on the model. Default None
"""
model = TestNet()
model__weight_initializer = None

"""
Data module
"""
datamodule = TestDataModule(50)

"""
Output

    description:
        Here we define the logger (Tensorboard, MLflow or another)
        That logger essentially tracks the experiment
        Some loggers can save the model and the model checkpoints as well (e.g. MLflow)
    
    variables:
        logger: The logger to use
        logger__log_config: Whether to log this config file. It's logger dependent
        logger__log_dataset: Whether to log the dataset (i.e. the datamodule name and the feature size). It's logger dependent
        logger__log_model: Whether to log the model at the end of the training through the logger. It's logger dependent
        logger__log_system_metrics: Whether to log system metrics. It's logger dependent
"""
logger = MLFlowLogger(
    experiment_name="config_test",
    tracking_uri="http://127.0.0.1:5000/",
    log_model=True,
)
logger__log_config = True
logger__log_dataset = True
logger__log_model = True
logger__log_system_metrics = True

"""
Tuner

    description:
        Options to tune your model (i.e. batch size finder and learning rate finder)
        
    variables:
        tuner__auto_scale_batch_size: Enabling auto-scaling of batch size to find the largest batch size that fits into memory. Large batch size often yields a better estimation of the gradients, but may also result in longer training time
        tuner__auto_scale_batch_size_mode: The mode to use for auto-scaling. Currently, this feature supports two modes 'power' scaling and 'binsearch' scaling
        tuner__auto_lr_find: Enabling the learning rate finder. Determines the best learning rate to start investigation. Replaces existing learning rate
"""
tuner__auto_scale_batch_size = False
tuner__auto_scale_batch_size_mode = "power"
tuner__auto_lr_find = False

"""
Trainer

    variables:
        trainer__devices: Specifying the accelerator parameter. Default 'auto'
        trainer__strategy: Training strategy to use. Default 'auto'
            If you use DistributedDataParallel (DDP), you have to set up a seed in order to have the same initial weight on each GPU
        trainer__num_nodes: Number of GPU nodes for distributed training. Default 1
        trainer__sync_batchnorm: In distributed training, if batchnorm is applied, setting it to True to assess batchnorm by taking into
            account all device (recommanded if you have small batch size). False to have a batchnorm by device
        trainer__precision: Precision to use for the training. Default 32
        trainer__gradient_clip_val: Setting infinity gradient to that value, by default we use 0
        trainer__reload_dataloaders_every_n_epoch: Reloading dataloader at n epochs. Default 0
        trainer__enable_model_summary: Printing a summary of the network
        trainer__profiler: Logging the time at the end ("simple", etc.)
        trainer__check_val_every_n_epoch: Applying validation every n epoch
        trainer__val_check_interval: Applying how often within one training epoch to check the validation set
        trainer__num_sanity_val_steps: Running n validation batches before starting the training routine for the sanity check, -1 for
            the whole validation. Default 2
        trainer__benchmark: Setting to True can increase the speed of your system if your input sizes don't change. However, if they do,
            then it might make your system slower. CUDNN auto-tuner will try to find the best algorithm for the hardware when a new input
            size is encountered
        trainer__deterministic: Setting to True to ensure reproducibility (ensure that CUDA uses a deterministic algorithm)
        trainer__fast_dev_run: DEBUGGING - Uncovering bugs without having to run your model. Set it to True to make some test
        trainer__overfit_batches: DEBUGGING - Overfitting on a same batch. If you don't have an overfitting, then you have a problem. Set
            1 for selecting a single batch and overfitting that batch. A float, for a percent of the training data. No shuffle
        trainer__enable_checkpointing: Enabling local checkpointing
        trainer__default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed
"""
trainer__accelerator = "gpu"
trainer__devices = 1
trainer__strategy = "auto"
trainer__num_nodes = 1
trainer__sync_batchnorm = False
trainer__min_epochs = 1
trainer__max_epochs = 20
trainer__precision = 32
trainer__gradient_clip_val = None
trainer__reload_dataloaders_every_n_epochs = 0
trainer__enable_model_summary = True
trainer__profiler = None
trainer__check_val_every_n_epoch = 1
trainer__val_check_interval = None
trainer__num_sanity_val_steps = 2
trainer__benchmark = None
trainer__deterministic = True
trainer__fast_dev_run = False
trainer__overfit_batches = 0.0
trainer__enable_checkpointing = True
trainer__default_root_dir = "../../output/model/Lightning"
